"""Translate sample ASR-style sentences with IndicTrans2 (verified official code) on the CPU."""
import os
import sys
import time

import torch
from sacremoses import MosesPunctNormalizer, MosesTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL = sys.argv[1]
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, trust_remote_code=True).eval()
norm, mtok = MosesPunctNormalizer(lang="en"), MosesTokenizer(lang="en")

SENTS = [
    "Hello, how are you?",
    "What is your name?",
    "I am going to the market tomorrow.",
    "Can you help me find the railway station?",
    "The weather is very nice today, let's go for a walk.",
    "I don't understand what you are saying.",
    "Please turn off the lights when you leave the room.",
    "My brother works as a software engineer in Bangalore.",
    "how much does this cost",
    "thank you very much for your help",
]


def prep(s):
    return "eng_Latn hin_Deva " + mtok.tokenize(norm.normalize(s.strip()), escape=False, return_str=True)


for beams in (1, 4):
    t = time.time()
    batch = tok([prep(s) for s in SENTS], padding="longest", return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**batch, num_beams=beams, max_length=128, use_cache=True)
    dec = tok.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print("== beams=%d  %.1f s" % (beams, time.time() - t))
    for s, h in zip(SENTS, dec):
        print("  %-55s -> %s" % (s, h))
