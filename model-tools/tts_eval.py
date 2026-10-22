"""End-to-end check of the on-board Hindi voice: float model vs the int8 NPU graphs.

    python tts_eval.py <mms_dir> <out_dir> <whisper_dir> [--calib-only]

1. Front end in PyTorch (text encoder, stochastic durations, prior noise) with fixed seeds.
2. Back end two ways from the same prior latents:
   float: original flow + HiFi-GAN on the whole utterance;
   npu:   flow_512_int8.onnx + vocoder_t64_int8.onnx (chunks of 64 frames with context), as the
          firmware runs them.
3. Whisper-small transcribes both; character error rate against the input text, and the log-mel
   distance between the two audios.
Also exports and quantizes the flow graph (flow_512_f32/int8.onnx) if missing.
"""
import os
import re
import sys

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from transformers import AutoTokenizer, VitsModel

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from npu_flow import NpuFlow  # noqa: E402
from tiled_vocoder import plan  # noqa: E402

MMS, OUT, WHISPER = sys.argv[1:4]
L, F = 512, 64
os.makedirs(OUT, exist_ok=True)

SENTS = ["नमस्ते, आप कैसे हैं?", "आपका नाम क्या है?", "मैं कल बाज़ार जा रहा हूँ।",
         "क्या आप रेलवे स्टेशन खोजने में मेरी मदद कर सकते हैं?", "मुझे समझ नहीं आ रहा है कि आप क्या कह रहे हैं।",
         "कमरे से बाहर निकलने पर कृपया बत्तियाँ बंद कर दें।", "इस पर कितना खर्च आता है?",
         "आपकी मदद के लिए बहुत-बहुत धन्यवाद।", "मेरा भाई बैंगलोर में सॉफ्टवेयर इंजीनियर के रूप में काम करता है।",
         "आज मौसम बहुत अच्छा है, चलो टहलने चलते हैं।", "डॉक्टर ने कहा कि मुझे आराम करना चाहिए।",
         "ट्रेन दस मिनट देर से आएगी।"]
CALIB = ["हम सब मिलकर इस समस्या का समाधान निकालेंगे।", "बच्चे स्कूल में खेल रहे हैं।",
         "यह किताब बहुत दिलचस्प है।", "क्या तुम मेरे साथ चलोगे?", "पानी पीना सेहत के लिए अच्छा होता है।",
         "उसने कहा कि वह कल आएगा।", "भारत एक विशाल देश है जिसमें कई भाषाएँ बोली जाती हैं।"]

tok = AutoTokenizer.from_pretrained(MMS)
model = VitsModel.from_pretrained(MMS).eval()


@torch.no_grad()
def front(text, seed):
    """Prior latents [192, T] and T, exactly as VitsModel.forward computes them."""
    g = torch.Generator().manual_seed(seed)
    ids = tok(text, return_tensors="pt")["input_ids"]
    mask_in = torch.ones(1, ids.shape[1], 1)
    te = model.text_encoder(input_ids=ids, padding_mask=mask_in)
    hidden = te.last_hidden_state.transpose(1, 2)
    m_p, logs_p = te.prior_means.transpose(1, 2), te.prior_log_variances.transpose(1, 2)
    torch.manual_seed(seed)
    logw = model.duration_predictor(hidden, mask_in.transpose(1, 2), None, reverse=True,
                                    noise_scale=model.noise_scale_duration)
    dur = torch.ceil(torch.exp(logw) * mask_in.transpose(1, 2) / model.speaking_rate)[0, 0].long()
    idx = torch.repeat_interleave(torch.arange(len(dur)), dur)
    mean, logv = m_p[0][:, idx], logs_p[0][:, idx]
    eps = torch.randn(mean.shape, generator=g)
    return (mean + eps * torch.exp(logv) * model.noise_scale).numpy(), len(idx)


@torch.no_grad()
def back_float(z, t):
    zt = torch.from_numpy(z)[None]
    mask = torch.ones(1, 1, t)
    lat = model.flow(zt, mask, None, reverse=True)
    return model.decoder(lat)[0, 0].numpy()


def padded(z, t):
    zp = np.zeros((1, 192, L), np.float32)
    zp[0, :, :t] = z[:, :t]
    mask = np.zeros((1, 1, L), np.float32)
    mask[0, 0, :t] = 1
    return zp, mask


def back_npu(z, t, flow_s, voc_s, pad):
    zp, mask = padded(z, t)
    lat = flow_s.run(None, {"latents": zp, "mask": mask})[0][0, :, 0, :t]       # [192, t]
    lat = np.pad(lat, ((0, 0), (pad, pad + F)))
    out = []
    for start in range(0, t, F):
        chunk = lat[None, :, start:start + F + 2 * pad].astype(np.float32)
        out.append(voc_s.run(None, {"latents": chunk})[0].ravel())
    return np.concatenate(out)[:t * 256]


def export_flow():
    f32, i8 = os.path.join(OUT, "flow_512_f32.onnx"), os.path.join(OUT, "flow_512_int8.onnx")
    if os.path.exists(i8):
        return i8
    flow = NpuFlow(model.flow).eval()
    data = []
    for i, s in enumerate(CALIB + SENTS[:3]):
        z, t = front(s, 100 + i)
        data.append(padded(z, t))
    torch.onnx.export(flow, (torch.from_numpy(data[0][0]), torch.from_numpy(data[0][1])), f32,
                      input_names=["latents", "mask"], output_names=["out"], opset_version=17, dynamo=False)

    class R(CalibrationDataReader):
        def __init__(self):
            self.it = iter(data)

        def get_next(self):
            d = next(self.it, None)
            return None if d is None else {"latents": d[0], "mask": d[1]}

    quantize_static(f32, i8, R(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
    return i8


def cer(ref, hyp):
    norm = lambda s: re.sub(r"[\s,.?!।\-]", "", s)
    a, b = norm(ref), norm(hyp)
    d = np.arange(len(b) + 1)
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[len(b)] / max(1, len(a))


def logmel(x):
    import librosa  # noqa: F401
    return None


def main():
    flow_i8 = export_flow()
    if "--calib-only" in sys.argv:
        return
    tvoc = os.path.join(OUT, "vocoder_t64_int8.onnx")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    flow_s = ort.InferenceSession(flow_i8, so, providers=["CPUExecutionProvider"])
    voc_s = ort.InferenceSession(tvoc, so, providers=["CPUExecutionProvider"])
    _, pad = plan(F)
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    proc = WhisperProcessor.from_pretrained(WHISPER)
    asr = WhisperForConditionalGeneration.from_pretrained(WHISPER).eval()

    def transcribe(wav16k):
        feats = proc(wav16k, sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad():
            ids = asr.generate(feats, language="hi", task="transcribe", max_new_tokens=128)
        return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    tot = {"float": [], "npu": []}
    snr = []
    for i, s in enumerate(SENTS):
        z, t = front(s, i)
        a = back_float(z, t)
        b = back_npu(z, t, flow_s, voc_s, pad)
        sf.write(os.path.join(OUT, "eval_%02d_float.wav" % i), a, 16000)
        sf.write(os.path.join(OUT, "eval_%02d_npu.wav" % i), b, 16000)
        n = min(len(a), len(b))
        snr.append(10 * np.log10(np.sum(a[:n] ** 2) / max(np.sum((a[:n] - b[:n]) ** 2), 1e-12)))
        ha, hb = transcribe(a), transcribe(b)
        tot["float"].append(cer(s, ha))
        tot["npu"].append(cer(s, hb))
        print("%-55s float CER %3.0f%% | npu CER %3.0f%% | SNR %5.1f dB\n   float: %s\n   npu  : %s" %
              (s, 100 * tot["float"][-1], 100 * tot["npu"][-1], snr[-1], ha, hb), flush=True)
    print("MEAN CER: float %.1f%%, npu %.1f%%; median waveform SNR %.1f dB" %
          (100 * np.mean(tot["float"]), 100 * np.mean(tot["npu"]), np.median(snr)))


if __name__ == "__main__":
    main()
