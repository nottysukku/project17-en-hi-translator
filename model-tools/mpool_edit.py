"""Write the ST Edge AI memory-pool file for one voice network, from models.json.

    python model-tools/mpool_edit.py <stm32n6.mpool> <vocoder|flow> <out.mpool>

Starts from the pool file the recognizer was compiled with (the owner's stm32n6.mpool) and
changes two pools only:
  octoFlash  offset/size = the network's flash slot in models.json (its weights go there)
  hyperRAM   offset/size = the voice_npu PSRAM region (its activations go there; the
             recognizer's activations at 0x90500000 stay untouched)
The internal NPU RAMs stay as they are: the networks never run at the same time (the
translator borrows the NPU from the recognizer).
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "firmware-n6", "src", "application", "translator", "models.json")


def pools(node):
    """Every dict in the file that describes a memory pool (has a 'name' and an 'offset')."""
    if isinstance(node, dict):
        if "name" in node and "offset" in node:
            yield node
        for v in node.values():
            yield from pools(v)
    elif isinstance(node, list):
        for v in node:
            yield from pools(v)


def set_pool(pool, offset, size):
    pool["offset"] = {"value": "0x%08X" % offset, "magnitude": "BYTES"}
    assert size % 1024 == 0
    pool["size"] = {"value": str(size // 1024), "magnitude": "KBYTES"}


def main():
    src, network, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    layout = json.load(open(MODELS, encoding="utf-8"))
    flash = {r["name"]: r for r in layout["flash"]}
    psram = {r["name"]: r for r in layout["psram_regions"]}
    if network not in ("vocoder", "flow"):
        sys.exit("network must be vocoder or flow")
    mpool = json.load(open(src, encoding="utf-8"))
    found = {p["name"]: p for p in pools(mpool)}
    if "octoFlash" not in found or "hyperRAM" not in found:
        sys.exit("No octoFlash/hyperRAM pool in %s; pools found: %s" % (src, sorted(found)))
    slot = flash[network]
    set_pool(found["octoFlash"], int(slot["address"], 0), int(slot["slot"], 0))
    npu = psram["voice_npu"]
    set_pool(found["hyperRAM"], int(npu["address"], 0), int(npu["size"], 0))
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        json.dump(mpool, f, indent=4)
    print("%s: octoFlash 0x%08X (%d KB), hyperRAM 0x%08X (%d KB)" %
          (dst, int(slot["address"], 0), int(slot["slot"], 0) // 1024, int(npu["address"], 0),
           int(npu["size"], 0) // 1024))


if __name__ == "__main__":
    main()
