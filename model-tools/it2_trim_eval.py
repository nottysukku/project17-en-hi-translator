"""Pack IndicTrans2 variants (vocabulary trims, decoder layers) and score each with the C engine.

    python it2_trim_eval.py <model_dir> <iitb_test.parquet> <it2_host.exe> <out_dir> [n] variant...

A variant is name:option,option,... with options
    src=N        keep N source embedding rows (it2_pack.py --src-keep)
    tgt=FILE     target vocabulary json (vocab_trim.py; default assets/hindi_vocab.json)
    dec=a.b.c    decoder layers to keep (default all 18)
for example  base:  s16k:src=16000  t31k:tgt=out/hindi_vocab_31k.json  d17:dec=0.1.2...16
The first variant is the reference for the "same translation" column. Variants are scored in
parallel (one C process each). Prints one line per variant: size, chrF++ and identical outputs.
"""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def parse(spec):
    name, _, opts = spec.partition(":")
    o = dict(src="0", tgt=os.path.join(HERE, "assets", "hindi_vocab.json"), dec=".".join(map(str, range(18))))
    for kv in filter(None, opts.split(",")):
        k, v = kv.split("=", 1)
        o[k] = v
    return name, o


def pack(model_dir, out_dir, name, o):
    blob = os.path.join(out_dir, "it2_%s.bin" % name)
    cmd = [PY, os.path.join(HERE, "it2_pack.py"), model_dir, o["tgt"], blob,
           "--dec-layers", o["dec"].replace(".", ","), "--src-keep", o["src"]]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return blob


def check(model_dir, test, exe, n, blob, save, compare):
    cmd = [PY, os.path.join(HERE, "it2_host_check.py"), model_dir, test, blob, exe, str(n), "--save", save, "--quiet"]
    if compare:
        cmd += ["--compare", compare]
    return subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8").stdout


def main():
    model_dir, test, exe, out_dir, n = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
    variants = [parse(s) for s in sys.argv[6:]]
    os.makedirs(out_dir, exist_ok=True)
    blobs = [pack(model_dir, out_dir, name, o) for name, o in variants]
    saves = [os.path.join(out_dir, "it2_%s_n%d.txt" % (name, n)) for name, _ in variants]
    ref = saves[0]
    first = check(model_dir, test, exe, n, blobs[0], ref, None)
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        jobs = [pool.submit(check, model_dir, test, exe, n, b, s, ref) for b, s in zip(blobs[1:], saves[1:])]
        for (name, o), blob, out in zip(variants, blobs, [first] + jobs):
            lines = (out if isinstance(out, str) else out.result()).splitlines()
            tok = next(l for l in lines if l.startswith("tokeniser"))
            chrf = next(l for l in lines if l.startswith("C engine")).split(";")[0].replace("C engine: ", "")
            same = next((l.split(": ")[1] for l in lines if l.startswith("same translation")), "-")
            print("%-8s %6.1f MB  %-58s | %s | same as %s: %s" %
                  (name, os.path.getsize(blob) / 1e6, chrf, tok, variants[0][0], same), flush=True)


if __name__ == "__main__":
    main()
