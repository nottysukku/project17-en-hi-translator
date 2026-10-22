"""Which IndicTrans2 target tokens does Hindi use? Count them on the IITB corpus's Hindi side.

    python vocab_trim.py <model_dir> <iitb_train.parquet> <out.json>
Writes the kept target ids (sorted), always including the special tokens and every
single-character Devanagari piece, so any Hindi output stays encodable.
"""
import json
import re
import sys
from collections import Counter

import pandas as pd
import sentencepiece as spm

sys.stdout.reconfigure(encoding="utf-8")
model_dir, parquet, out = sys.argv[1:4]
sp = spm.SentencePieceProcessor(model_file=model_dir + "/model.TGT")
vocab = json.load(open(model_dir + "/dict.TGT.json", encoding="utf-8"))
d = pd.read_parquet(parquet)
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
kept = sorted(set(i for i, n in ids.items() if n >= 2) | set(single) | set(special))
json.dump(kept, open(out, "w"))
print("kept %d ids -> %s" % (len(kept), out))
