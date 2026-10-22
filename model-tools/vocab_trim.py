"""Which IndicTrans2 target tokens does Hindi use? Count them on the IITB corpus's Hindi side.

    python vocab_trim.py <model_dir> <iitb_train.parquet> <out.json> [--max N]

Writes the kept target ids (sorted), always including the special tokens and every
single-character Devanagari piece, so any Hindi output stays encodable. Without --max it keeps
every token seen at least twice; with --max N it keeps the N most frequent (specials and single
characters included). The counts go to <out>.counts.json.
"""
import argparse
import json
import re
import sys
from collections import Counter

import pandas as pd
import sentencepiece as spm

sys.stdout.reconfigure(encoding="utf-8")
ap = argparse.ArgumentParser()
ap.add_argument("model_dir")
ap.add_argument("parquet")
ap.add_argument("out")
ap.add_argument("--max", type=int, default=0, help="keep at most N tokens (0: every token seen twice)")
args = ap.parse_args()
sp = spm.SentencePieceProcessor(model_file=args.model_dir + "/model.TGT")
vocab = json.load(open(args.model_dir + "/dict.TGT.json", encoding="utf-8"))
d = pd.read_parquet(args.parquet)
hi = [r["hi"] for r in d["translation"]]
print("sentences:", len(hi))
counts = Counter()
for i in range(0, len(hi), 20000):
    for pieces in sp.encode(hi[i:i + 20000], out_type=str):
        counts.update(pieces)
ids = Counter()
for piece, n in counts.items():
    if piece in vocab:
        ids[vocab[piece]] += n
total = sum(ids.values())
for min_count in (1, 2, 5, 10, 50):
    kept = [i for i, n in ids.items() if n >= min_count]
    cov = sum(ids[i] for i in kept) / total
    print("min count %3d: %6d ids, token coverage %.5f" % (min_count, len(kept), cov))
single = [i for t, i in vocab.items() if re.fullmatch(r"▁?[ऀ-ॿ]", t)]
special = [vocab[t] for t in ("<s>", "<pad>", "</s>", "<unk>") if t in vocab]
must = set(single) | set(special)
if args.max:
    ranked = [i for i, n in ids.most_common() if n >= 2 and i not in must]
    kept = sorted(must | set(ranked[:max(0, args.max - len(must))]))
else:
    kept = sorted(set(i for i, n in ids.items() if n >= 2) | must)
cov = sum(ids[i] for i in kept) / total
json.dump(kept, open(args.out, "w"))
json.dump({str(i): n for i, n in ids.items()}, open(args.out.rsplit(".", 1)[0] + ".counts.json", "w"))
print("kept %d ids (token coverage %.5f) -> %s" % (len(kept), cov, args.out))
