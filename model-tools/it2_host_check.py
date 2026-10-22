"""Check the C engine (it2_host.exe) against the Python reference: tokeniser ids and translations.

    python it2_host_check.py <model_dir> <iitb_test.parquet> <blob.bin> <it2_host.exe> [n]
"""
import os
import subprocess
import sys

import sacrebleu

sys.stdout.reconfigure(encoding="utf-8")
model_dir, test, blob, exe = sys.argv[1:5]
N = int(sys.argv[5]) if len(sys.argv) > 5 else 128
sys.argv = [sys.argv[0], model_dir, test, str(N)]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from it2_ref import IT2  # noqa: E402
from it2_variants import CONV, subset  # noqa: E402
EXTRA = ["I'm going, aren't you?", "don't stop.", "It's John's book.", "we'll see. They've gone!",
         "I can't, won't, shouldn't.", "what's up?", "students' books are here", "hello   world"]
rows = subset() + CONV + [(e, None) for e in EXTRA]
inp = os.path.join(os.path.dirname(blob), "it2_check_in.txt")
open(inp, "w", encoding="utf-8", newline="\n").write("\n".join(r[0] for r in rows) + "\n")

ref = IT2(model_dir)
c_ids = subprocess.run([exe, blob, inp, "--ids"], capture_output=True, text=True, encoding="utf-8").stdout.splitlines()
bad = 0
for (en, _), line in zip(rows, c_ids):
    py = ref.encode_text(en)
    c = [int(x) for x in line.split()]
    if py != c:
        bad += 1
        print("TOKENISER DIFF: %r\n  py %s\n  c  %s" % (en, py, c))
print("tokeniser: %d of %d sentences identical" % (len(rows) - bad, len(rows)))

out = subprocess.run([exe, blob, inp], capture_output=True, text=True, encoding="utf-8").stdout.splitlines()
hyps = [l.split("\t")[0] for l in out]
times = [(int(l.split("\t")[1]), int(l.split("\t")[2]), int(l.split("\t")[3]), int(l.split("\t")[4])) for l in out]
scored = [(h, hi) for h, (_, hi) in zip(hyps, rows) if hi is not None]
chrf = sacrebleu.corpus_chrf([h for h, _ in scored], [[hi for _, hi in scored]], word_order=2).score
print("C engine: chrF++ %.1f on %d sentences (float reference 48.1); mean %.0f src / %.0f tgt tokens, "
      "%.0f ms encode, %.0f ms decode per sentence on this PC" %
      (chrf, len(scored), sum(t[0] for t in times) / len(times), sum(t[1] for t in times) / len(times),
       sum(t[2] for t in times) / len(times), sum(t[3] for t in times) / len(times)))
for (en, _), h in list(zip(rows, hyps))[-16:]:
    print("  %-50s %s" % (en, h))
