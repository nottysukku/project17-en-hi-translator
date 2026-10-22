"""Export the MMS-TTS Hindi vocoder (HiFi-GAN) for the STM32N6 NPU and quantize it to int8.

    python tts_export.py <mms_dir> <out_dir> [frames]

1. Synthesize calibration sentences with the full float model and keep the latents that feed
   the vocoder (192 channels x frames, 62.5 frames/s).
2. Export the vocoder for a fixed chunk of `frames` latent frames (the firmware overlaps chunks).
3. Quantize it statically to int8 (QDQ, per-channel weights) on those latents.
4. Report how close the int8 chunk output is to float, per chunk, as SNR.
"""
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType, quantize_static)
from transformers import AutoTokenizer, VitsModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from npu_vocoder import NpuVocoder  # noqa: E402
from npu_vocoder_w import NpuVocoderW  # noqa: E402
from tiled_vocoder import TiledVocoder  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
MMS, OUT = sys.argv[1], sys.argv[2]
FRAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 60
LAYOUT = sys.argv[4] if len(sys.argv) > 4 else 'w'
os.makedirs(OUT, exist_ok=True)

CALIB = [
    "नमस्ते, आप कैसे हैं?", "मुझे समझ में नहीं आ रहा है कि आप क्या कह रहे हैं।", "मैं कल बाज़ार जा रहा हूँ।",
    "क्या आप रेलवे स्टेशन ढूंढने में मेरी मदद कर सकते हैं?", "आज मौसम बहुत अच्छा है, चलो टहलने चलते हैं।",
    "जब आप कमरे से बाहर जाएं तो कृपया लाइट बंद कर दें।", "मेरा भाई बैंगलोर में सॉफ्टवेयर इंजीनियर के रूप में काम करता है।",
    "इसकी कीमत कितनी है?", "आपकी मदद के लिए बहुत-बहुत धन्यवाद।", "आपका नाम क्या है?",
    "हम सब मिलकर इस समस्या का समाधान निकालेंगे।", "बच्चे स्कूल में खेल रहे हैं।",
    "डॉक्टर ने कहा कि मुझे आराम करना चाहिए।", "ट्रेन दस मिनट देर से आएगी।",
]


class Vocoder(torch.nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, latents):
        return self.dec(latents)


def latents_of(model, tok, text, seed):
    torch.manual_seed(seed)
    ids = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**ids)
    return out.spectrogram[0].numpy()  # [192, frames]


def chunks(lat):
    t = lat.shape[1]
    for start in range(0, max(1, t - FRAMES + 1), FRAMES // 2):
        c = lat[:, start:start + FRAMES]
        if c.shape[1] < FRAMES:
            c = np.pad(c, ((0, 0), (0, FRAMES - c.shape[1])))
        yield c[None].astype(np.float32)


class Reader(CalibrationDataReader):
    def __init__(self, data):
        self.it = iter(data)

    def get_next(self):
        x = next(self.it, None)
        return None if x is None else {"latents": x}


def main():
    tok = AutoTokenizer.from_pretrained(MMS)
    model = VitsModel.from_pretrained(MMS).eval()
    data = [c for i, s in enumerate(CALIB) for c in chunks(latents_of(model, tok, s, i))]
    print("calibration chunks: %d of %d frames" % (len(data), FRAMES))

    f32 = os.path.join(OUT, "vocoder_%s%d_f32.onnx" % (LAYOUT, FRAMES))
    if LAYOUT == 't':
        voc = TiledVocoder(model.decoder, model.config.leaky_relu_slope, FRAMES).eval()
        pad = voc.pad
        data = []  # chunks with their real latent context, zero-padded at the utterance ends
        for i, s in enumerate(CALIB):
            lat = np.pad(latents_of(model, tok, s, i), ((0, 0), (pad, pad + FRAMES)))
            for start in range(0, lat.shape[1] - 2 * pad - FRAMES + 1, FRAMES // 2):
                data.append(lat[None, :, start:start + FRAMES + 2 * pad].astype(np.float32))
    else:
        voc = (NpuVocoderW if LAYOUT == 'w' else NpuVocoder)(model.decoder, model.config.leaky_relu_slope).eval()
    torch.onnx.export(voc, torch.from_numpy(data[0]), f32, input_names=["latents"],
                      output_names=["audio"], opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(f32))
    i8 = os.path.join(OUT, "vocoder_%s%d_int8.onnx" % (LAYOUT, FRAMES))
    quantize_static(f32, i8, Reader(data[::2]), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    extra_options={"ActivationSymmetric": False, "WeightSymmetric": True})

    s32 = ort.InferenceSession(f32, providers=["CPUExecutionProvider"])
    s8 = ort.InferenceSession(i8, providers=["CPUExecutionProvider"])
    snrs = []
    for x in data[1::2]:
        a = s32.run(None, {"latents": x})[0].ravel()
        b = s8.run(None, {"latents": x})[0].ravel()
        snrs.append(10 * np.log10(np.sum(a ** 2) / max(np.sum((a - b) ** 2), 1e-12)))
    print("int8 vs float per-chunk SNR: median %.1f dB, worst %.1f dB" % (np.median(snrs), np.min(snrs)))
    np.save(os.path.join(OUT, "vocoder_%s%d_calib.npy" % (LAYOUT, FRAMES)), np.concatenate(data[:8]))
    print(f32, i8)


if __name__ == "__main__":
    main()
