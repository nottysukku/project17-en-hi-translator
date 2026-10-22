"""How much does quantizing the TTS text encoder cost? Float vs 8-bit vs 4-bit weights.

    python tts_front_eval.py <mms_dir> <whisper_dir> [out_dir]

Each variant runs tts_front_ref.Front (int8 activations per 64-group when quantized, like the C
code) with the same noise; the float flow and HiFi-GAN make the audio. Reports per variant: the
share of input ids whose duration equals the float one, the total length change, the SNR of the
prior means against float, and Whisper-small's character error rate on the audio.
"""
import json
import os
import re
import sys

import numpy as np
import soundfile as sf
import torch
from transformers import VitsModel, WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts_front_ref import Front, text_to_ids  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

# The test sentences and the CER of tts_eval.py (which loads its models on import).
SENTS = ["नमस्ते, आप कैसे हैं?", "आपका नाम क्या है?", "मैं कल बाज़ार जा रहा हूँ।",
         "क्या आप रेलवे स्टेशन खोजने में मेरी मदद कर सकते हैं?", "मुझे समझ नहीं आ रहा है कि आप क्या कह रहे हैं।",
         "कमरे से बाहर निकलने पर कृपया बत्तियाँ बंद कर दें।", "इस पर कितना खर्च आता है?",
         "आपकी मदद के लिए बहुत-बहुत धन्यवाद।", "मेरा भाई बैंगलोर में सॉफ्टवेयर इंजीनियर के रूप में काम करता है।",
         "आज मौसम बहुत अच्छा है, चलो टहलने चलते हैं।", "डॉक्टर ने कहा कि मुझे आराम करना चाहिए।",
         "ट्रेन दस मिनट देर से आएगी।"]


def cer(ref, hyp):
    norm = lambda s: re.sub(r"[\s,.?!।\-]", "", s)
    a, b = norm(ref), norm(hyp)
    d = np.arange(len(b) + 1)
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[len(b)] / max(1, len(a))


def main():
    mms, whisper = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else None
    model = VitsModel.from_pretrained(mms).eval()
    vocab = json.load(open(os.path.join(mms, "vocab.json"), encoding="utf-8"))
    proc = WhisperProcessor.from_pretrained(whisper)
    asr = WhisperForConditionalGeneration.from_pretrained(whisper).eval()

    def transcribe(wav):
        feats = proc(wav, sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad():
            ids = asr.generate(feats, language="hi", task="transcribe", max_new_tokens=128)
        return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    variants = {"float": Front(model.state_dict()), "w8": Front(model.state_dict(), 8),
                "w4": Front(model.state_dict(), 4)}
    stats = {k: dict(same=[], length=[], snr=[], cer=[]) for k in variants}
    for i, text in enumerate(SENTS):
        ids = text_to_ids(text, vocab)
        g = torch.Generator().manual_seed(100 + i)
        nd = torch.randn(2, len(ids), generator=g)
        big = torch.randn(192, 4000, generator=g)
        ref = None
        line = "%-45s" % text[:45]
        for name, front in variants.items():
            dur, z, inter = front.run(ids, nd, lambda n: big[:, :n])
            with torch.no_grad():
                lat = model.flow(z[None], torch.ones(1, 1, z.shape[1]), None, reverse=True)
                wav = model.decoder(lat)[0, 0].numpy()
            if out:
                os.makedirs(out, exist_ok=True)
                sf.write(os.path.join(out, "front_%02d_%s.wav" % (i, name)), wav, 16000)
            if ref is None:
                ref = (dur, inter["m"])
            st = stats[name]
            st["same"].append(float((dur == ref[0]).float().mean()))
            st["length"].append(float(dur.sum()) / float(ref[0].sum()) - 1)
            err = float(((inter["m"] - ref[1]) ** 2).sum())
            st["snr"].append(10 * np.log10(float((ref[1] ** 2).sum()) / max(err, 1e-20)))
            st["cer"].append(cer(text, transcribe(wav)))
            line += " | %s CER %3.0f%%" % (name, 100 * st["cer"][-1])
        print(line, flush=True)
    for name, st in stats.items():
        print("%-6s durations same %5.1f%%  length change %+5.1f%%  prior-mean SNR %5.1f dB  CER %5.1f%%" %
              (name, 100 * np.mean(st["same"]), 100 * np.mean(np.abs(st["length"])),
               np.median(st["snr"]) if name != "float" else float("inf"), 100 * np.mean(st["cer"])))


if __name__ == "__main__":
    main()
