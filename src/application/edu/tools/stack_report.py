"""Worst-case stack of a thread: GCC's per-function frames (-fstack-usage) along the deepest call chain.

  python stack_report.py BUILD_DIR [--root display_thread_entry] [--root edu_math_parse] ...

BUILD_DIR is an EDU build (-DAPP_EDU_MODE=ON writes a .su file per object). Calls are read from
the disassembly of BUILD_DIR/src/application/app.elf (bl, and tail calls b.w to a function start).
Functions without a .su file (newlib) get their frame from their prologue (push / vpush / sub sp).
Calls through function pointers are listed in INDIRECT below; any other indirect call is reported.
Recursion is allowed only for the functions in RECURSION, with their proven depth bound.
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys

# Calls through a function pointer: caller -> the functions it can reach that way.
INDIRECT = {
    "BSP_TS_GetState": ["GT911_GetState"],
    "BSP_TS_Init": ["GT911_Init", "GT911_GetCapabilities"],
    "GT911_ReadID": ["ReadRegWrap"],
    "GT911_GetState": ["ReadRegWrap"],
    "GT911_Init": ["ReadRegWrap", "WriteRegWrap"],
    "GT911_RegisterBusIO": ["BSP_I2C2_Init"],
    "ReadRegWrap": ["BSP_I2C2_ReadReg16"],
    "WriteRegWrap": ["BSP_I2C2_WriteReg16"],
    "gt911_read_reg": ["ReadRegWrap"],
    "gt911_write_reg": ["WriteRegWrap"],
    "mw_lcd_present": ["present_frame"],
    "logger_log": ["drv_uart_write", "drv_uart_tx_ready"],
    "logger_print": ["drv_uart_write", "drv_uart_tx_ready"],
    "format_record": ["drv_uart_write", "drv_uart_tx_ready"],
    "wait_output_ready": ["drv_uart_tx_ready"],
    "_printf_common": ["__ssputs_r"],
    "_printf_i": ["__ssputs_r"],
    # the semaphore-put notify callback: tx_semaphore_put_notify() is never called in this firmware
    "_tx_semaphore_ceiling_put": [],
}
# Recursive functions and their depth bound: enumerate() recurses once per number of a reading,
# at most MAX_NUMBERS (3) times, so it is on the stack at most 4 times.
RECURSION = {"enumerate": 4}
# Exception entry on the thread's stack (extended frame with FP context, 26 words + alignment)
# and ThreadX's save of the callee-saved core and FP registers when the thread is preempted.
CONTEXT_BYTES = 108 + 104

FUNC_RE = re.compile(r"^([0-9a-f]+) <([^>]+)>:$")
INSN_RE = re.compile(r"^\s*([0-9a-f]+):\s+(\S+)\s*(.*)$")
TARGET_RE = re.compile(r"\b([0-9a-f]+) <([^>+]+)>\s*$")


def key(name):
    return re.sub(r"(\.\d+)+$", "", name)


def read_su(build):
    frames = {}
    dynamic = set()
    for path in glob.glob(os.path.join(build, "**", "*.su"), recursive=True):
        for line in open(path, encoding="utf-8", errors="replace"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = key(parts[0].rsplit(":", 1)[-1])
            frames[name] = max(frames.get(name, 0), int(parts[1]))
            if "dynamic" in parts[2]:
                dynamic.add(name)
    return frames, dynamic


def regs(text):
    n = 0
    for part in text.strip("{} ").split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            n += int(b[1:]) - int(a[1:]) + 1
        elif part:
            n += 1
    return n


def disassemble(elf, objdump):
    out = subprocess.run([objdump, "-d", "--no-show-raw-insn", elf], capture_output=True, text=True, check=True).stdout
    funcs = {}
    current = None
    for line in out.splitlines():
        m = FUNC_RE.match(line)
        if m:
            current = key(m.group(2))
            funcs.setdefault(current, {"calls": set(), "tails": set(), "indirect": False, "prologue": [], "start": int(m.group(1), 16)})
            continue
        m = INSN_RE.match(line)
        if not m or current is None:
            continue
        op, args = m.group(2), m.group(3)
        f = funcs[current]
        if len(f["prologue"]) < 8:
            f["prologue"].append((op, args))
        t = TARGET_RE.search(args)
        if op in ("bl", "blx") and t:
            f["calls"].add(key(t.group(2)))
        elif op in ("blx",) and not t:
            f["indirect"] = True
        elif op.startswith("b") and t and op not in ("bl", "blx", "bx", "bic", "bics", "bfi", "bfc"):
            target = key(t.group(2))
            if target != current:
                f["tails"].add(target)
    return funcs


def prologue_frame(f):
    total = 0
    for op, args in f["prologue"]:
        if op in ("push", "push.w", "stmdb") and ("sp!" in args or op.startswith("push")):
            total += 4 * regs(args[args.index("{"):])
        elif op == "vpush":
            r = args[args.index("{"):]
            total += (8 if "d" in r else 4) * regs(r)
        elif op in ("sub", "sub.w", "subw") and args.startswith("sp, ") and "#" in args:
            total += int(args.split("#")[1].split()[0], 0)
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("build")
    ap.add_argument("--root", action="append")
    ap.add_argument("--objdump", default=shutil.which("arm-none-eabi-objdump") or "arm-none-eabi-objdump")
    args = ap.parse_args()
    # edu_math_parse_many is not linked into the firmware (the board runs the quiz only).
    roots = args.root or ["display_thread_entry", "edu_display_poll", "edu_display_start", "edu_math_parse",
                          "edu_math_may_continue", "draw_screen", "BSP_TS_GetState", "publish_text"]
    frames, dynamic = read_su(args.build)
    funcs = disassemble(os.path.join(args.build, "src", "application", "app.elf"), args.objdump)

    estimated = set()

    def frame(name):
        if name in frames:
            return frames[name]
        if name in funcs:
            estimated.add(name)
            return prologue_frame(funcs[name])
        return 0

    memo = {}
    unresolved = set()

    def cost(name, stack):
        if name in memo:
            return memo[name]
        if name in stack:
            return None  # recursion: handled by the caller below
        f = funcs.get(name, {"calls": set(), "tails": set(), "indirect": False})
        callees = set(f["calls"]) | set(INDIRECT.get(name, []))
        if f["indirect"] and name not in INDIRECT:  # an empty list marks a pointer known to be unused
            unresolved.add(name)
        best_call, best_path = 0, []
        recursive = False
        for c in sorted(callees):
            if c == name:
                recursive = True
                continue
            r = cost(c, stack + [name])
            if r is None:
                sys.exit("unexpected recursion through %s -> %s" % (name, c))
            if r[0] > best_call:
                best_call, best_path = r
        own = frame(name)
        if recursive:
            if name not in RECURSION:
                sys.exit("recursive function without a depth bound: %s" % name)
            own *= RECURSION[name]
        total, path = own + best_call, [(name, own)] + best_path
        for c in sorted(f["tails"]):
            r = cost(c, stack + [name])
            if r and r[0] > total:
                total, path = r
        memo[name] = (total, path)
        return memo[name]

    print("Stack use, deepest call chain (bytes; frames from -fstack-usage, * = from the prologue):\n")
    for root in roots:
        total, path = cost(root, [])
        print("%-24s %5d B" % (root, total))
        for name, own in path:
            mark = "*" if name in estimated else " "
            rec = " (x%d recursion)" % RECURSION[name] if name in RECURSION else ""
            print("    %5d %s %s%s" % (own, mark, name, rec))
        print()
    print("Interrupt / preemption on the thread's stack: +%d B (FP exception frame, ThreadX context save)" % CONTEXT_BYTES)
    worst = cost(roots[0], [])[0] + CONTEXT_BYTES
    print("%s worst case including that: %d B" % (roots[0], worst))
    dyn = sorted(n for n in dynamic if n in memo)
    if dyn:
        print("WARNING: dynamic (variable-size) frames on these paths: %s" % ", ".join(dyn))
    if unresolved:
        print("WARNING: indirect calls not resolved (add them to INDIRECT): %s" % ", ".join(sorted(unresolved)))


if __name__ == "__main__":
    main()
