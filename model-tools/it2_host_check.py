"""Check the C engine (it2_host.exe) against the Python reference: tokeniser ids and translations.

    python it2_host_check.py <model_dir> <iitb_test.parquet> <blob.bin> <it2_host.exe> [n]
                             [--save out.tsv] [--compare base.tsv] [--quiet]

The tokeniser check follows the blob: if it2_pack.py trimmed the source vocabulary (--src-keep),
the Python side encodes with the same restricted SentencePiece vocabulary and maps the ids to the
blob's embedding rows. --save writes the C translations; --compare reports how many sentences
are translated exactly as in an earlier --save (for example the untrimmed model).
"""
import argparse
import json
import os
import subprocess
import sys

import sacrebleu

sys.stdout.reconfigure(encoding="utf-8")
ap = argparse.ArgumentParser()
ap.add_argument("model_dir")
ap.add_argument("test")
ap.add_argument("blob")
ap.add_argument("exe")
ap.add_argument("n", nargs="?", type=int, default=128)
ap.add_argument("--save")
ap.add_argument("--compare")
ap.add_argument("--quiet", action="store_true", help="print only the summary lines")
args = ap.parse_args()
sys.argv = [sys.argv[0], args.model_dir, args.test, str(args.n)]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from it2_ref import IT2, restricted_spm  # noqa: E402
from it2_variants import CONV, subset  # noqa: E402
EXTRA = ["I'm going, aren't you?", "don't stop.", "It's John's book.", "we'll see. They've gone!",
         "I can't, won't, shouldn't.", "what's up?", "students' books are here", "hello   world"]
rows = subset() + CONV + [(e, None) for e in EXTRA]
inp = os.path.splitext(args.blob)[0] + "_check_in.txt"
open(inp, "w", encoding="utf-8", newline="\n").write("\n".join(r[0] for r in rows) + "\n")

ref = IT2(args.model_dir)
meta_path = args.blob + ".json"
meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
row_of = None
if "src_rows" in meta and len(meta["src_rows"]) < len(ref.src_vocab):
    # Trimmed source vocabulary: SentencePiece splits the dropped pieces the way it2.c does.
    row_of = {orig: r for r, orig in enumerate(meta["src_rows"])}
    ref.src_spm = restricted_spm(os.path.join(args.model_dir, "model.SRC"),
                                 lambda p: p not in ref.src_vocab or ref.src_vocab[p] in row_of)
unk_row = row_of[ref.src_vocab["<unk>"]] if row_of else None

c_ids = subprocess.run([args.exe, args.blob, inp, "--ids"], capture_output=True, text=True,
                       encoding="utf-8").stdout.splitlines()
bad = 0
for (en, _), line in zip(rows, c_ids):
    py = ref.encode_text(en)
    if row_of:
        py = [row_of.get(i, unk_row) for i in py]
    c = [int(x) for x in line.split()]
    if py != c:
        bad += 1
        if not args.quiet:
            print("TOKENISER DIFF: %r\n  py %s\n  c  %s" % (en, py, c))
print("tokeniser: %d of %d sentences identical" % (len(rows) - bad, len(rows)))

out = subprocess.run([args.exe, args.blob, inp], capture_output=True, text=True, encoding="utf-8").stdout.splitlines()
hyps = [l.split("\t")[0] for l in out]
times = [tuple(int(x) for x in l.split("\t")[1:5]) for l in out]
scored = [(h, hi) for h, (_, hi) in zip(hyps, rows) if hi is not None]
chrf = sacrebleu.corpus_chrf([h for h, _ in scored], [[hi for _, hi in scored]], word_order=2).score
print("C engine: chrF++ %.1f on %d sentences (float reference 48.1 on 120); mean %.0f src / %.0f tgt tokens, "
      "%.0f ms encode, %.0f ms decode per sentence on this PC" %
      (chrf, len(scored), sum(t[0] for t in times) / len(times), sum(t[1] for t in times) / len(times),
       sum(t[2] for t in times) / len(times), sum(t[3] for t in times) / len(times)))
if args.save:
    open(args.save, "w", encoding="utf-8", newline="\n").write("\n".join(hyps) + "\n")
if args.compare:
    base = open(args.compare, encoding="utf-8").read().splitlines()
    same = sum(a == b for a, b in zip(hyps, base))
    print("same translation as %s: %d of %d sentences (%.1f%%)" %
          (os.path.basename(args.compare), same, len(hyps), 100.0 * same / len(hyps)))
if not args.quiet:
    for (en, _), h in list(zip(rows, hyps))[-16:]:
        print("  %-50s %s" % (en, h))
