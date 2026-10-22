"""Check the C checker (edu_math.c) against the Python one (mathspeech.py) on the PC.

  python run_host_test.py [--game-dir ASR_Math_Game] [--out DIR] [--zig PATH] [--fuzz N]

1. Fails if edu_math_tables.h was not made from the current mathspeech.py (run tools/gen_edu_tables.py).
2. Writes the test vectors (gen_vectors.py) into DIR (default: <firmware>/build_edu_host).
3. Compiles edu_math.c + host_test.c for the PC with Zig (zig cc) and runs it: 100 % agreement needed.
4. Compiles and runs game_test.c: the quiz flow (edu_game.c) with a simulated clock.
Exit code 0 only when both pass.

Zig is found from --zig, $ZIG, PATH, or the ziglang pip package (python -m pip install ziglang).
"""
import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EDU = os.path.dirname(HERE)
FIRMWARE = os.path.normpath(os.path.join(EDU, "..", "..", ".."))
DEFAULT_GAME = os.path.normpath(os.path.join(FIRMWARE, "..", "..", "..", "ASR_Math_Game"))


def find_zig(explicit):
    for cand in (explicit, os.environ.get("ZIG"), shutil.which("zig")):
        if cand and os.path.exists(cand):
            return [cand]
    try:
        import ziglang  # noqa: F401
        return [sys.executable, "-m", "ziglang"]
    except ImportError:
        pass
    sys.exit("Zig not found: pass --zig, set ZIG, or `python -m pip install ziglang`")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--game-dir", default=DEFAULT_GAME)
    ap.add_argument("--out", default=os.path.join(FIRMWARE, "build_edu_host"))
    ap.add_argument("--zig", default=None)
    ap.add_argument("--fuzz", type=int, default=40000)
    ap.add_argument("--show", type=int, default=25, help="mismatches to print")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    mathspeech = os.path.join(args.game_dir, "mathspeech.py")

    print("[1/4] word tables")
    subprocess.check_call([sys.executable, os.path.join(EDU, "tools", "gen_edu_tables.py"), "--check",
                           "--mathspeech", mathspeech])

    print("[2/4] test vectors from mathspeech.py")
    vectors = os.path.join(args.out, "vectors.tsv")
    subprocess.check_call([sys.executable, os.path.join(HERE, "gen_vectors.py"), vectors,
                           "--game-dir", args.game_dir, "--fuzz", str(args.fuzz)])

    zig = find_zig(args.zig)
    flags = ["cc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-I", EDU]
    suffix = ".exe" if os.name == "nt" else ""

    print("[3/4] checker: host build (zig cc), every vector against Python")
    exe = os.path.join(args.out, "host_test" + suffix)
    subprocess.check_call(zig + flags + [os.path.join(EDU, "edu_math.c"), os.path.join(HERE, "host_test.c"), "-o", exe])
    vectors_ok = subprocess.call([exe, vectors, str(args.show)]) == 0

    print("[4/4] quiz flow (edu_game.c)")
    exe = os.path.join(args.out, "game_test" + suffix)
    subprocess.check_call(zig + flags + [os.path.join(EDU, "edu_math.c"), os.path.join(EDU, "edu_game.c"),
                                         os.path.join(HERE, "game_test.c"), "-o", exe])
    game_ok = subprocess.call([exe]) == 0
    sys.exit(0 if (vectors_ok and game_ok) else 1)


if __name__ == "__main__":
    main()
