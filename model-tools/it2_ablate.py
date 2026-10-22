"""Ablation on the same IITB subset: decoder layers, 4-bit weights, Hindi-only output vocabulary."""
import copy
import json
import os
import sys

import sacrebleu
import torch

model_dir, test, vocab_json = sys.argv[1:4]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 120
sys.argv = [sys.argv[0], model_dir, test, str(N)]
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from it2_ref import IT2  # noqa: E402
from it2_variants import CONV, quantized_weights, subset  # noqa: E402

rows = subset() + CONV
refs = [hi for _, hi in rows if hi is not None]
keep = torch.tensor(json.load(open(vocab_json)))
keep = keep[(keep != 0) & (keep != 1) & (keep != 3)]


def score(name, m, allowed=None):
    hyps = [m.decode_ids(m.greedy(m.encode_text(en), allowed=allowed)) for en, _ in rows]
    chrf = sacrebleu.corpus_chrf([h for h, (_, hi) in zip(hyps, rows) if hi is not None], [refs], word_order=2).score
    print("%-40s chrF++ %.1f" % (name, chrf), flush=True)


D10 = [0, 2, 4, 6, 8, 10, 12, 14, 16, 17]
base = IT2(model_dir)
score("float, 18 dec", base)
score("float, 10 dec", IT2(model_dir, dec_layers=D10, weights=base.w))
qw = quantized_weights(base.w, 4, 4, 64)
q10 = IT2(model_dir, dec_layers=D10, weights=qw)
q10.lm_head = qw["lm_head.weight"]
score("int4 g64, 10 dec", q10)
score("int4 g64, 10 dec, Hindi vocab", q10, keep)
q18 = IT2(model_dir, weights=qw)
q18.lm_head = qw["lm_head.weight"]
score("int4 g64, 18 dec, Hindi vocab", q18, keep)
