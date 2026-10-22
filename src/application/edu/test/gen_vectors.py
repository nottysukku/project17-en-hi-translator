"""Test vectors for the C checker (edu_math.c): what mathspeech.py says about each text.

  python gen_vectors.py OUT.tsv [--game-dir ASR_Math_Game] [--fuzz 40000] [--jobs 8]

Covers every case of test_mathspeech.py (all table lines 2-30 x 1-15, every phrasing, true and wrong
answers, quiz and free mode, the misspelt and recital tests), the recognizer outputs in
ASR_Math_Game/eval/*.jsonl and every word-by-word prefix of them (the board judges text as it
arrives), and random word soup built from the checker's own vocabulary.

One line per vector, tab-separated; "-" is None:
  P cat ranges expect_a expect_b text | template a b c correct ambiguous doubt score span0 span1 shown
  M cat ranges table text | items count {a,b,c,correct,ambiguous,score,span0,span1;}...
  C cat text | may_continue
ranges is "-" (defaults) or "tmin,tmax,mmin,mmax". template: 0 none 1 full 2 answer 3 partial 4 question.
"""
import argparse
import glob
import json
import multiprocessing
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GAME = os.path.normpath(os.path.join(HERE, *[".."] * 7, "ASR_Math_Game"))
TEMPLATES = {"none": 0, "full": 1, "answer": 2, "partial": 3, "question": 4}

ms = None
tm = None


def init(game_dir):
    global ms, tm
    if game_dir not in sys.path:
        sys.path.insert(0, game_dir)
    import mathspeech
    import test_mathspeech
    ms, tm = mathspeech, test_mathspeech


def opt(v):
    return "-" if v is None else str(int(v))


def ranges_field(r):
    return "-" if r is None else "%d,%d,%d,%d" % r


def make_ranges(r):
    return None if r is None else ms.Ranges(*r)


def run(job):
    """job -> output line."""
    kind = job[0]
    if kind == "P":
        _, cat, r, expect, text = job
        c = ms.parse(text, expect, make_ranges(r))
        shown = c.shown.replace("×", "x")
        ea, eb = expect if expect else (None, None)
        return "\t".join(["P", cat, ranges_field(r), opt(ea), opt(eb), text, "|", str(TEMPLATES[c.template]),
                          opt(c.a), opt(c.b), opt(c.c), opt(c.correct), str(int(c.ambiguous)),
                          str(int(bool(c.doubt))), str(int(c.score)), str(c.span[0]), str(c.span[1]), shown])
    if kind == "M":
        _, cat, r, table, text = job
        claims, items = ms.parse_many(text, table=table, ranges=make_ranges(r))
        body = ";".join("%s,%s,%s,%s,%d,%d,%d,%d" % (opt(c.a), opt(c.b), opt(c.c), opt(c.correct), int(c.ambiguous),
                                                     int(c.score), c.span[0], c.span[1]) for c in claims)
        return "\t".join(["M", cat, ranges_field(r), opt(table), text, "|", str(items), str(len(claims)), body])
    _, cat, text = job
    return "\t".join(["C", cat, text, "|", str(int(ms.may_continue(text)))])


# ------------------------------------------------------------------ cases

def suite_jobs():
    """Every case of test_mathspeech.py."""
    jobs = []
    say, phrase, wrong = ms.say, tm.phrase, tm.wrong_answers
    for a in tm.TABLES:
        for b in tm.MULTS:
            answers = [a * b] + wrong(a, b)
            for kind in tm.EXPLICIT:
                for c in answers:
                    t = phrase(kind, a, b, c)
                    jobs.append(("P", "explicit_free", None, None, t))
                    jobs.append(("P", "explicit_quiz", None, (a, b), t))
            for kind in ("bare", "bare_coll"):
                for c in answers:
                    t = phrase(kind, a, b, c)
                    jobs.append(("P", "bare_quiz", None, (a, b), t))
                    jobs.append(("P", "bare_free", None, None, t))
            for c in answers:
                for t in (say(c), say(c, "colloq"), say(c, "and"), "IS " + say(c), "UM " + say(c)):
                    jobs.append(("P", "answer_quiz", None, (a, b), t))
    examples = [
        ("TWENTY THREE TIMES EIGHT IS ONE HUNDRED EIGHTY FOUR", None),
        ("TWENTY THREE EIGHT JA ONE HUNDRED EIGHTY FOUR", None),
        ("THIRTY FOURTEEN FOUR TWENTY", None), ("TWENTY ONE'S ARE TWENTY", None),
        ("TWENTY TIMES ONE IS TWENTY", None), ("ONE EIGHTY FOUR", (23, 8)),
        ("THIRTY X FOURTEEN IS FOUR HUNDRED TWENTY", None),
        ("TWENTY TREE TIMES ATE IS ONE HUNDRED EIGHTY FOR", None),
        ("TWENTI THREE TIME EIGHT IS ONE HUNDERED EIGHTY FOUR", None),
        ("TWENTYTHREE TIMES EIGHT IS ONEHUNDRED EIGHTYFOUR", None),
        ("SEV EN TIMES FIF TEEN IS ONE HUNDRED FIVE", None),
        ("TIRTY TIMES FORTEEN IS FOR HUNDRED TWENTY", None),
        ("NINETEEN TIMES TOO IS THIRTY EIGHT", None),
        ("TWELVE TIMES TWELVE EQUALS TO ONE FOURTY FOUR", None),
        ("TWENTY NINE TIMES FIFTEEN IS FOUR THIRTY FIVE", None),
        ("TWENTY THREE EIGHTH ARE ONE EIGHTY FOUR", None),
        ("TWENTY THREE TIMES EIGHT IS ONE HUNDRED EIGHTY FIVE", (23, 8)),
        ("TWENTY THREE TIMES EIGHT IS ONE EIGHTY", (23, 8)),
        ("TWENTY THREE TIMES EIGHT IS ONE HUNDRED FORTY EIGHT", (23, 8)),
        ("THREE TIMES FIVE IS FIFTY", (3, 5)),
        ("TWENTY THREE TIMES NINE IS TWO HUNDRED SEVEN", (23, 8)),
        ("TWENTY THREE TIMES EIGHT IS", None), ("", None), ("HELLO THERE BOARD", None),
        ("FOUR TIMES FIFTEEN IS SIXTEE", None), ("SIXTEE", (4, 15)), ("SIXTEE", (4, 4)),
        ("FOUR TIMES FOUR IS FIFTEE", None),
        ("ONE EIGHTY FIVE NO SORRY ONE EIGHTY FOUR", (23, 8)),
        # typed input: digits and symbols
        ("23 x 8 = 184", None), ("23 × 8 = 184", None), ("23*8=184", None), ("twenty three times eight is 184", (23, 8)),
        ("184", (23, 8)), ("0", (2, 1)), ("007 x 3 = 21", None), ("ONE’S", None), ("TWENTY ONE’S ARE TWENTY ONE", None),
    ]
    for text, expect in examples:
        jobs.append(("P", "examples", None, expect, text))
        jobs.append(("P", "examples", None, None, text))
        jobs.append(("C", "examples", text))
    # the noisy report, exactly as the test draws it, then more seeds in quiz mode
    for seed, rounds, quiz in ((0, 3, False), (1, 2, True), (2, 2, False), (3, 2, True)):
        rng = random.Random(seed)
        for _ in range(rounds):
            for a in tm.TABLES:
                for b in tm.MULTS:
                    for kind in ("times", "times_coll", "plural", "ja"):
                        for c in (a * b, a * b + 1):
                            t = tm.noisy(phrase(kind, a, b, c), rng)
                            jobs.append(("P", "noisy_quiz" if quiz else "noisy", None, (a, b) if quiz else None, t))
    # recitals, exactly as test_recital_every_table draws them
    rng = random.Random(1)
    for n in tm.TABLES:
        for kind in ("plural", "times", "ja"):
            wrong_lines = {rng.randrange(1, 16), rng.randrange(1, 16)}
            t = tm.recital(n, kind, wrong_lines, rng)
            jobs.append(("M", "recital", None, n, t))
            jobs.append(("M", "recital_notable", None, None, t))
            jobs.append(("P", "recital_parse", None, None, t))
            jobs.append(("P", "recital_parse", None, (n, 7), t))
    return jobs


def eval_jobs(game_dir):
    jobs = []
    for path in sorted(glob.glob(os.path.join(game_dir, "eval", "*.jsonl"))):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            heard, a, b = r["heard"], r["a"], r["b"]
            jobs.append(("P", "eval_free", None, None, heard))
            jobs.append(("P", "eval_quiz", None, (a, b), heard))
            jobs.append(("C", "eval", heard))
            jobs.append(("M", "eval_many", None, a, heard))
            words = heard.split()
            for k in range(1, len(words)):
                prefix = " ".join(words[:k])
                jobs.append(("P", "eval_prefix", None, (a, b), prefix))
                jobs.append(("C", "eval_prefix", prefix))
    return jobs


def fuzz_jobs(count, seed=7):
    """Random utterances from the checker's own vocabulary, misspelt, glued, split and blurred."""
    rng = random.Random(seed)
    lex = ms.DEFAULT_LEXICON
    known = sorted(set(lex.exact) | ms.FILLERS | ms.CORRECTIONS | {"A", "OH", "O", "BY"})
    aliases = sorted(ms.ALIASES)
    numbers = sorted(ms.NUMBER_WORDS) + ["HUNDRED"]
    blurred = [w[:-3] + "TEE" for w in ms.TEENS if w.endswith("TEEN")] + ["FOURTEE", "FORTEE", "EIGHTEE", "NINTEE"]
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def misspell(w):
        if len(w) < 2:
            return w
        k = rng.randrange(len(w))
        op = rng.random()
        if op < 0.4:
            return w[:k] + w[k + 1:]
        if op < 0.7:
            return w[:k] + rng.choice(letters) + w[k + 1:]
        if op < 0.85:
            return w[:k] + rng.choice(letters) + w[k:]
        return w[:k] + w[k] + w[k:]  # doubled letter

    def word():
        r = rng.random()
        if r < 0.40:
            return rng.choice(numbers)
        if r < 0.52:
            return rng.choice(known)
        if r < 0.64:
            return rng.choice(aliases)
        if r < 0.76:
            return misspell(rng.choice(numbers + ["TIMES", "EQUALS", "INTO", "MULTIPLIED", "ARE"]))
        if r < 0.80:
            return rng.choice(blurred)
        if r < 0.86:
            return rng.choice(numbers) + rng.choice(numbers + ["TIMES", "IS", "ARE", "AND"])  # glued
        if r < 0.90:
            w = rng.choice(numbers + ["TIMES", "HUNDRED"])
            k = rng.randrange(1, len(w)) if len(w) > 1 else 1
            return w[:k] + " " + w[k:]  # split
        if r < 0.94:
            return "".join(rng.choice(letters) for _ in range(rng.randrange(1, 6)))
        if r < 0.96:
            return str(rng.randrange(0, 1000))
        if r < 0.98:
            return rng.choice(["×", "*", "=", "'", "’", ",", ".", "?", "-", "ONE'S", "TWO’S"])
        return rng.choice(["NO", "SORRY", "WAIT", "ACTUALLY"])

    def structured():
        a, b = rng.randrange(1, 32), rng.randrange(0, 17)
        c = a * b + rng.choice([0, 0, 0, 1, -1, 10, -10, a, 100])
        c = max(c, 0)
        style = rng.choice(["std", "colloq", "and", "digits"])
        parts = [ms.say(a), rng.choice(["TIMES", "X", "INTO", "MULTIPLIED BY", ""]),
                 ms.say(b) if rng.random() < 0.7 else ms.say_plural(b), rng.choice(["IS", "ARE", "EQUALS", "JA", "", "IS EQUAL TO"]),
                 ms.say(c, style)]
        text = " ".join(p for p in parts if p)
        out = []
        for w in text.split():
            r = rng.random()
            out.append(misspell(w) if r < 0.12 else rng.choice(blurred) if r < 0.14 else w)
        if rng.random() < 0.2 and len(out) > 2:
            k = rng.randrange(len(out) - 1)
            out[k:k + 2] = [out[k] + out[k + 1]]
        if rng.random() < 0.15:
            out.insert(rng.randrange(len(out) + 1), word())
        return " ".join(out), (a, b)

    jobs = []
    for i in range(count):
        if rng.random() < 0.5:
            text, ab = structured()
        else:
            text = " ".join(word() for _ in range(rng.randrange(0, 15)))
            ab = (rng.randrange(2, 31), rng.randrange(1, 16))
        if rng.random() < 0.3:
            text = text.lower()
        r = None
        if rng.random() < 0.15:
            tmin = rng.randrange(1, 10)
            mmin = rng.randrange(0, 5)
            r = (tmin, tmin + rng.randrange(0, 25), mmin, mmin + rng.randrange(0, 15))
        expect = None if rng.random() < 0.4 else (ab if rng.random() < 0.7 else (rng.randrange(1, 31), rng.randrange(1, 16)))
        if expect is not None and min(expect) < 1:
            expect = None  # mathspeech divides by an operand: a 0 question raises (the game never asks one)
        jobs.append(("P", "fuzz", r, expect, text))
        if i % 3 == 0:
            jobs.append(("C", "fuzz", text))
        if i % 4 == 0:
            jobs.append(("M", "fuzz", r, rng.choice([None, ab[0], rng.randrange(2, 31)]), text))
    return jobs


def safe(job):
    """Drop texts the C code is not meant to match: tabs/newlines, digits of 1000 or more."""
    text = job[-1]
    if any(ch in text for ch in "\t\r\n|"):
        return False
    import re
    return all(int(m) < 1000 for m in re.findall(r"\d+", text))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--game-dir", default=DEFAULT_GAME)
    ap.add_argument("--fuzz", type=int, default=40000)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    init(args.game_dir)
    jobs = suite_jobs() + eval_jobs(args.game_dir) + fuzz_jobs(args.fuzz)
    jobs = [j for j in jobs if safe(j)]
    with multiprocessing.Pool(args.jobs, initializer=init, initargs=(args.game_dir,)) as pool:
        lines = pool.map(run, jobs, chunksize=500)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line + "\n")
    counts = {}
    for j in jobs:
        counts[j[1]] = counts.get(j[1], 0) + 1
    print("wrote %d vectors to %s" % (len(lines), args.out))
    for cat, n in sorted(counts.items()):
        print("  %-16s %7d" % (cat, n))


if __name__ == "__main__":
    main()
