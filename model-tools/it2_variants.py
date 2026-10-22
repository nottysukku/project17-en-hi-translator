"""How much quality do on-device compromises cost? IndicTrans2 variants on an IITB test subset.

Variants: float (reference), weights int8 per row, int4 in groups of 64, decoder layers dropped
(no fine-tuning), and output restricted to Devanagari tokens. Scores: chrF++ against the human
references and the share of sentences identical to the float model's output.

    python it2_variants.py <model_dir> <iitb_test.parquet> [n_sentences]
"""
import copy
import os
import re
import sys
import time

import pandas as pd
import sacrebleu
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from it2_ref import IT2  # noqa: E402

MODEL, TEST = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 150

CONV = [("Hello, how are you?", None), ("What is your name?", None), ("I am going to the market tomorrow.", None),
        ("Can you help me find the railway station?", None), ("I don't understand what you are saying.", None),
        ("Please turn off the lights when you leave the room.", None), ("How much does this cost?", None),
        ("Thank you very much for your help.", None)]


def subset():
    d = pd.read_parquet(TEST)
    rows = [(r["en"], r["hi"]) for r in d["translation"]]
    rows = [r for r in rows if 4 <= len(r[0].split()) <= 20 and not re.search(r"[0-9%$/@]", r[0])]
    step = max(1, len(rows) // N)
    return rows[::step][:N]


def quantize(w, bits, group=None):
    """Symmetric fake-quantization of a 2-D weight, per row or per group of columns."""
    qmax = 2 ** (bits - 1) - 1
    if group:
        r, c = w.shape
        g = w.view(r, c // group, group)
        s = g.abs().amax(dim=2, keepdim=True).clamp_min(1e-8) / qmax
        return (torch.round(g / s).clamp(-qmax, qmax) * s).view(r, c)
    s = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
    return torch.round(w / s).clamp(-qmax, qmax) * s


def quantized_weights(base, enc_bits, dec_bits, group=None):
    w = {}
    for k, v in base.items():
        if v.dim() == 2 and k.endswith("weight"):
            bits = enc_bits if k.startswith("model.encoder") else dec_bits
            w[k] = quantize(v, bits, group if bits == 4 else None) if bits else v
        else:
            w[k] = v
    w["lm_head.weight"] = w["model.decoder.embed_tokens.weight"]
    return w


def devanagari_ids(model):
    ok = re.compile(r"^[▁ऀ-ॿ -@[-`{-~‌‍]+$")
    ids = sorted(i for t, i in model.tgt_vocab.items() if ok.match(t) and re.search(r"[ऀ-ॿ]|^[▁]?[0-9.,?!:;'\"()-]+$", t))
    return ids + [2]


def run(name, model, rows, allowed=None, base_out=None):
    t = time.time()
    hyps = []
    for en, _ in rows:
        ids = model.encode_text(en)
        hyps.append(model.decode_ids(model.greedy(ids, allowed=allowed)))
    refs = [hi for _, hi in rows if hi is not None]
    chrf = sacrebleu.corpus_chrf([h for h, (_, hi) in zip(hyps, rows) if hi is not None], [refs], word_order=2).score
    same = sum(a == b for a, b in zip(hyps, base_out)) / len(hyps) if base_out else 1.0
    print("%-34s chrF++ %5.1f  same-as-float %5.1f%%  %.2f s/sent" % (name, chrf, 100 * same, (time.time() - t) / len(rows)),
          flush=True)
    return hyps


def main():
    rows = subset() + CONV
    base = IT2(MODEL)
    print("%d test sentences; Devanagari-only target vocab: %d of %d" %
          (len(rows), len(devanagari_ids(base)), len(base.tgt_vocab)))
    ref_out = run("float", base, rows)
    allowed = torch.tensor(devanagari_ids(base))
    run("float, Devanagari vocab only", base, rows, allowed, ref_out)
    for name, eb, db, g in (("int8 rows, enc+dec", 8, 8, None), ("int4 g64, enc+dec", 4, 4, 64),
                            ("enc int8, dec int4 g64", 8, 4, 64), ("enc int4 g32, dec int8", 4, 8, 32)):
        m = copy.copy(base)
        m.w = quantized_weights(base.w, eb, db, g)
        m.lm_head = m.w["lm_head.weight"]
        run(name, m, rows, None, ref_out)
    for layers in ([0, 2, 4, 6, 8, 10, 12, 14, 16, 17], [0, 3, 6, 9, 12, 15, 17], [0, 6, 12, 17]):
        m = IT2(MODEL, dec_layers=layers, weights=base.w)
        run("decoder %d layers (no fine-tune)" % len(layers), m, rows[:40], None, ref_out[:40])
    for (en, _), h in zip(rows[-8:], ref_out[-8:]):
        print("  %-50s %s" % (en, h))


if __name__ == "__main__":
    main()
