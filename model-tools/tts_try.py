"""Synthesize Hindi sentences with MMS-TTS (transformers' built-in VitsModel) and time it."""
import os
import sys
import time

import soundfile as sf
import torch
from transformers import AutoTokenizer, VitsModel

sys.stdout.reconfigure(encoding="utf-8")
MODEL, OUT = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(MODEL)
model = VitsModel.from_pretrained(MODEL).eval()
print("params: %.1f M" % (sum(p.numel() for p in model.parameters()) / 1e6))
for name, mod in model.named_children():
    print("  %-22s %.2f M" % (name, sum(p.numel() for p in mod.parameters()) / 1e6))
SENTS = ["नमस्ते, आप कैसे हैं?", "मुझे समझ में नहीं आ रहा है कि आप क्या कह रहे हैं।",
         "मैं कल बाज़ार जा रहा हूँ।", "क्या आप रेलवे स्टेशन ढूंढने में मेरी मदद कर सकते हैं?"]
torch.manual_seed(0)
for i, s in enumerate(SENTS):
    ids = tok(s, return_tensors="pt")
    t = time.time()
    with torch.no_grad():
        wav = model(**ids).waveform[0].numpy()
    dt = time.time() - t
    sf.write(os.path.join(OUT, "mms_%d.wav" % i), wav, 16000)
    print("%d chars -> %d ids, %.2f s audio in %.2f s: %s" % (len(s), ids["input_ids"].shape[1], len(wav) / 16000, dt, s))
