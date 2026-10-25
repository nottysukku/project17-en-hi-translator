"""Build asr_indian_english.ipynb from asr_lib.py and the cells below.

    python make_notebook.py            writes asr_indian_english.ipynb next to this file
    python make_notebook.py --script   also writes notebook_cells.py: the code cells as one script,
                                       for a quick test outside Colab (ASR_QUICK_TEST=1)

The library is the tested code (smoke_test.py); the notebook writes it to /content/asr_lib.py in
its second cell, so the notebook needs nothing from this repository.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

INTRO = r"""
# Project 17: teach the DK's speech recognizer Indian-accented English

**Version 2 (2026-10-25).** Version 1 lost the model's ReLUs when it read the file (they are inside
its int8 steps), so the old model scored 100% and the losses were `nan`. If you ran version 1: open
this one and Run all. It trains into new folders on Drive (`checkpoints_v2`), and reuses the
downloaded index and check sets. Cell 8 now stops before training if the old model does not score
sensibly.

This notebook fine-tunes the recognizer that runs on the STM32N6570-DK (CynipsNet, the model in
`firmware-n6/src/asr_engine/models/cynipsnet_punct1`) on English spoken with Indian accents, then
gives back a file for ST Edge AI. The new model has **exactly the same layers and size** (7.2 M
parameters, 7.0 MB of weights at the same flash address), so it fits the DK where the old one is.

**What it learns from** (downloaded from Hugging Face by the notebook, about 47 GB):

| Data | Hours | What it is |
|---|---|---|
| Common Voice, Indian accent | 164 | People across India reading English sentences (with punctuation) |
| NPTEL lectures | ~170 of ~900 | Indian professors teaching in English |
| IndicTTS English | ~30 | Indian speakers of 13 languages reading English |
| LibriSpeech (kept as a reminder) | 100 per epoch | The original training data, so it does not forget it |

It is checked on held-out Common Voice speakers, NPTEL test lectures, LibriSpeech dev-clean and,
if you give it a Hugging Face token, **Svarah** (9.6 h, 117 Indian speakers from 65 districts).

**Before you start (once):**
1. Copy `cynipsnet_punct1.onnx` from your PC to Google Drive, folder `project17_asr` (create it):
   `C:\Users\choprasu\Downloads\Reverse_engineering2\Reverse_Engineering\cynipsnet_re\deploy\out\cynipsnet_punct1\cynipsnet_punct1.onnx`
   (the file ST Edge AI compiled for the board; its path is in the model's report).
2. Optional, for the Svarah test: open https://huggingface.co/datasets/ai4bharat/Svarah, log in,
   accept its terms; make a token at https://huggingface.co/settings/tokens (read access); in
   Colab click the key icon on the left, add a secret named `HF_TOKEN` with the token, and allow
   this notebook to use it.
3. Runtime > Change runtime type > **T4 GPU**.

**Then:** Runtime > Run all. The first session downloads and indexes the data (about 30 min),
measures the old model, and starts training. Training saves to Drive every 15 minutes and stops by
itself before Colab's session limit. **When a session ends, open the notebook again and Run all:
it continues where it stopped** (the data is downloaded again, the index is reused).

**How long:** it prints its speed. On a T4 expect roughly 1.5 h per epoch of ~460 h of audio,
so the default 16 epochs take about 25 GPU hours: well inside your 250 hours. The original model's
README says "268 of 268 epochs are pure hardware": those are NPU scheduling steps on the board,
not training passes, so there is no need to match that number. The original fine-tuning stopped at
training epoch 4 (its file was `epoch_04_step_00021311_v9.tflite`).

**At the end** the last cells compare the old and new model the way the board runs them (same
features, same 6-second windows, int8) and write `export/cynipsnet_inen1/` on Drive. The final
text cell says what to run on your PC.
"""

SETUP = r"""
# 1. GPU, Google Drive, packages
import os, subprocess, sys
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout)
    from google.colab import drive
    drive.mount("/content/drive")
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "onnx", "onnxruntime", "huggingface_hub",
                    "soundfile"], check=True)
DRIVE = os.environ.get("ASR_DRIVE", "/content/drive/MyDrive/project17_asr")  # survives the session
WORK = os.environ.get("ASR_WORK", "/content/work")                           # fast local disk
LIB = os.environ.get("ASR_LIB_DIR", "/content")
os.makedirs(DRIVE, exist_ok=True)
os.makedirs(WORK, exist_ok=True)
print("Drive folder:", DRIVE)
"""

SETTINGS = r"""
# 3. Settings: the data, the training, the names
sys.path.insert(0, LIB)
import importlib, json, pickle, shutil, time, urllib.request
import numpy as np, torch
import asr_lib as A
importlib.reload(A)

QUICK = bool(os.environ.get("ASR_QUICK_TEST"))  # a few minutes on a PC, to test the notebook itself
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get("HF_TOKEN")
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN")
START_ONNX = os.environ.get("ASR_START_ONNX", DRIVE + "/cynipsnet_punct1.onnx")
NAME = "cynipsnet_inen1"  # the new model's folder name in the firmware

S = (lambda n: 1) if QUICK else (lambda n: n)  # quick test: one file per source
SOURCES = [
    A.Source("cv_in", "ishands/commonvoice-indian_accent", ["data/train-*.parquet"], "sentence", punct=True,
             shards=S(0), dur_col="duration_ms", dur_scale=0.001, speaker_col="client_id"),
    A.Source("nptel", "skbose/indian-english-nptel-v0", ["data/train-*.parquet"], "transcription", punct=False,
             shards=S(40)),
    A.Source("indictts", "SPRINGLab/IndicTTS-English", ["data/train-*.parquet"], "text", punct=False,
             shards=S(4)),
    A.Source("libri_clean", "openslr/librispeech_asr", ["all/train.clean.100/*.parquet"], "text", punct=False,
             shards=S(0), replay=True, hours_per_epoch=60, lspc=True, id_col="id"),
    A.Source("libri_other", "openslr/librispeech_asr", ["all/train.other.500/*.parquet"], "text", punct=False,
             shards=S(10), replay=True, hours_per_epoch=40, lspc=True, id_col="id"),
]
if QUICK:
    SOURCES = [s for s in SOURCES if s.name != "indictts"]  # 2.8 GB per file: too big for the PC test
DEV_SOURCES = {
    "nptel_test": A.Source("nptel_test", "skbose/indian-english-nptel-v0", ["data/test-*.parquet"],
                           "transcription", punct=False, shards=1),
    "libri_dev": A.Source("libri_dev", "openslr/librispeech_asr", ["all/validation.clean/*.parquet"], "text",
                          punct=False, lspc=True, id_col="id"),
}
SVARAH = A.Source("svarah", "ai4bharat/Svarah", ["data/test-*.parquet"], "text", punct=False,
                  audio_col="audio_filepath", gated=True)
DEV_N = 6 if QUICK else 400     # utterances per check set
BOARD_N = 3 if QUICK else 40    # utterances per set for the board simulation at the end (slow: CPU)

CFG = A.TrainConfig(
    lr=2e-4,              # peak learning rate (fine-tuning: low)
    epochs=1 if QUICK else 16,
    batch_seconds=40 if QUICK else 480,
    kl_weight=0.5,        # how strongly LibriSpeech rows keep the old model's answers
    session_hours=(1.5 / 60) if QUICK else 11.0,  # Colab Pro sessions last about 12 h (free: lower this)
    eval_every=4 if QUICK else 2000,
    log_every=2 if QUICK else 100,
    amp=torch.cuda.is_available(),
    workers=2 if QUICK else 0,
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", DEVICE, "| quick test" if QUICK else "", "| Hugging Face token:", "yes" if HF_TOKEN else "no")
"""

CHECK_MODEL = r"""
# 4. The starting model: the one the board runs now
assert os.path.exists(START_ONNX), "Upload cynipsnet_punct1.onnx to " + DRIVE + " first (see the top)."
net = A.OnnxNet(START_ONNX)
relus = sum(1 for op in net.ops if op[0] == "Relu")
print("input", net.in_dims, "output", net.out_dims, "parameters %d" % A.count_params(net))
print("ReLUs: %d (%d of them inside int8 steps); CynipsNet has 33" % (relus, len(net.qdq_relu)))
assert net.out_dims[net.out_class] == len(A.TOKENS), "not the 32-class punctuation model"
del net
"""

DOWNLOAD = r"""
# 5. Download the data to the local disk (every session; about 47 GB, 10-20 minutes)
print("free disk: %.0f GB" % (shutil.disk_usage(WORK).free / 1e9))
FILES = {}
for s in SOURCES + list(DEV_SOURCES.values()):
    t0 = time.time()
    FILES[s.name] = A.download(s, WORK + "/hf", HF_TOKEN)
    print("%-12s %3d files, %.0f s" % (s.name, len(FILES[s.name]), time.time() - t0))
if HF_TOKEN:
    try:
        FILES["svarah"] = A.download(SVARAH, WORK + "/hf", HF_TOKEN)
    except Exception as e:
        print("Svarah not available (accept its terms on Hugging Face):", type(e).__name__)
LSPC = WORK + "/librispeech_pc_manifests.tar.gz"  # LibriSpeech with punctuation (openslr.org/145)
if not os.path.exists(LSPC):
    urllib.request.urlretrieve("https://www.openslr.org/resources/145/manifests.tar.gz", LSPC)
print("free disk now: %.0f GB" % (shutil.disk_usage(WORK).free / 1e9))
"""

INDEX = r"""
# 6. Index: which rows to use, their text and length (built once, kept on Drive)
UNITS = {}
idx_dir = DRIVE + ("/index_quick" if QUICK else "/index")
for s in SOURCES:
    UNITS[s.name] = A.build_index(s, FILES[s.name], "%s/%s.json" % (idx_dir, s.name), LSPC)
    print("%-12s %7.1f h  %7d utterances" % (s.name, sum(u[2] for u in UNITS[s.name]),
                                          sum(len(u[3]) for u in UNITS[s.name])))
"""

DEV = r"""
# 7. Check sets (built once, kept on Drive)
dev_cache = DRIVE + ("/dev_sets_quick.pkl" if QUICK else "/dev_sets.pkl")
if os.path.exists(dev_cache):
    DEV_SETS = pickle.load(open(dev_cache, "rb"))
else:
    lspc = A.load_lspc(LSPC)
    DEV_SETS = {
        "cv_in_dev": A.dev_set(SOURCES[0], FILES["cv_in"], DEV_N),  # speakers never trained on
        "nptel_test": A.dev_set(DEV_SOURCES["nptel_test"], FILES["nptel_test"], DEV_N, split_speakers=False),
        "libri_dev_clean": A.dev_set(DEV_SOURCES["libri_dev"], FILES["libri_dev"], DEV_N, lspc=lspc,
                                     split_speakers=False),
    }
    pickle.dump(DEV_SETS, open(dev_cache, "wb"))
for k, v in DEV_SETS.items():
    print("%-16s %4d utterances, %.1f min" % (k, len(v), sum(len(p) for p, _ in v) / A.SR / 60))
"""

BASELINE = r"""
# 8. The old model on the check sets (once), then resume if a checkpoint exists
# (v2: the first version lost the model's ReLUs; its checkpoints and baseline are not used)
CK = DRIVE + ("/checkpoints_quick" if QUICK else "/checkpoints_v2")
trainer = A.Trainer(START_ONNX, CK, CFG, DEVICE)
base_path = DRIVE + ("/baseline_quick.json" if QUICK else "/baseline_v2.json")
if os.path.exists(base_path):
    BASE = json.load(open(base_path))
else:
    BASE = trainer.evaluate(DEV_SETS)
    json.dump(BASE, open(base_path, "w"))
print("old model:", "  ".join("%s WER %.2f%%" % (k, 100 * v["wer"]) for k, v in BASE.items()))
for pcm, ref in DEV_SETS["libri_dev_clean"][:3]:
    print("  said: ", ref)
    print("  heard:", trainer.transcribe([pcm])[0])
if not QUICK and BASE["libri_dev_clean"]["wer"] > 0.30:
    raise RuntimeError("The old model should score about 7%% on LibriSpeech dev-clean, not %.0f%%: the model "
                       "file was not read correctly. Do not train; send the output of cells 4 and 8."
                       % (100 * BASE["libri_dev_clean"]["wer"]))
# the same model as the board runs it: int8, 6 s windows, statistics of the last 30 s (CPU, a few minutes)
board_path = DRIVE + ("/baseline_board_quick.json" if QUICK else "/baseline_board_v2.json")
if not os.path.exists(board_path):
    n = 3 if QUICK else 25
    json.dump({"wer": A.BoardSim(START_ONNX).evaluate(DEV_SETS["libri_dev_clean"][:n])["wer"], "n": n},
              open(board_path, "w"))
BOARD0 = json.load(open(board_path))
print("old model as the board runs it, LibriSpeech dev-clean (%d sentences in a row): WER %.2f%%"
      % (BOARD0["n"], 100 * BOARD0["wer"]))
if trainer.load("last.pt"):
    print("resuming: epoch %d, step %d, %.0f h of audio so far" % (trainer.epoch, trainer.step, trainer.audio_hours))
"""

TRAIN = r"""
# 9. Train. Stops by itself before the session limit; run the notebook again to continue.
def log(msg):
    print(msg, flush=True)
    with open(DRIVE + ("/train_log_quick.txt" if QUICK else "/train_log_v2.txt"), "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M ") + msg + "\n")

trainer.fit(UNITS, SOURCES, DEV_SETS, log)
"""

RESULTS = r"""
# 10. Progress so far: word error rate on each check set, old model first
rows = [("old model", {k: v["wer"] for k, v in BASE.items()})] + \
       [("epoch %d step %d" % (h["epoch"], h["step"]), h) for h in trainer.history]
names = list(DEV_SETS)
print("%-22s" % "" + "".join("%18s" % n for n in names))
for label, r in rows:
    print("%-22s" % label + "".join("%17.2f%%" % (100 * r[n]) if n in r else "%18s" % "-" for n in names))
done = trainer.epoch >= CFG.epochs
print("\ntraining finished: run the next cells" if done else "\ntraining not finished: run the notebook again")
"""

EXPORT = r"""
# 11. Export the best model: float ONNX -> int8 with the board's input scale -> network_shape.h
best = A.Trainer(START_ONNX, CK, CFG, DEVICE)
assert best.load("best.pt"), "no best.pt yet"
EXPORT_DIR = DRIVE + ("/export_quick/" if QUICK else "/export/") + NAME
os.makedirs(EXPORT_DIR, exist_ok=True)
float_onnx = best.model.export_float_onnx(EXPORT_DIR + "/%s_float.onnx" % NAME)
calib = A.calibration_windows(sum((v[: max(2, DEV_N // 4)] for v in DEV_SETS.values()), []), 400)
in_scale, out_scale = A.quantize_int8(float_onnx, EXPORT_DIR + "/%s.onnx" % NAME, calib)
A.write_network_shape(EXPORT_DIR + "/network_shape.h", NAME, in_scale, out_scale)
print("int8 model: input scale %.10f (the board's), output scale %.6f" % (in_scale, out_scale))
"""

BOARD = r"""
# 12. Old vs new, run as the board runs them (int8, 6 s windows, the decoder's steps). CPU: slow.
old_sim, new_sim = A.BoardSim(START_ONNX), A.BoardSim(EXPORT_DIR + "/%s.onnx" % NAME)
sets = {k: v[:BOARD_N] for k, v in DEV_SETS.items()}
if "svarah" in FILES:
    sets["svarah_test"] = A.dev_set(SVARAH, FILES["svarah"], 4 if QUICK else 300, split_speakers=False)
TABLE = {}
for k, items in sets.items():
    t0 = time.time()
    TABLE[k] = {"old": old_sim.evaluate(items)["wer"], "new": new_sim.evaluate(items)["wer"], "n": len(items)}
    print("%-16s old %6.2f%%   new %6.2f%%   (%d utterances, %.0f s)" %
          (k, 100 * TABLE[k]["old"], 100 * TABLE[k]["new"], len(items), time.time() - t0))
prov = {"name": NAME, "start": os.path.basename(START_ONNX), "tokens": A.TOKENS,
        "steps": best.step, "epochs": best.epoch, "audio_hours": round(best.audio_hours, 1),
        "sources": [s.name for s in SOURCES], "train_config": vars(CFG),
        "check_sets_float": {k: v["wer"] for k, v in BASE.items()}, "history": best.history,
        "board_simulation_int8_wer": TABLE, "input_scale": in_scale, "output_scale": out_scale,
        "date": time.strftime("%Y-%m-%d")}
json.dump(prov, open(EXPORT_DIR + "/provenance.json", "w"), indent=1)
print("written:", sorted(os.listdir(EXPORT_DIR)))
"""

PC_STEPS = r"""
## 13. On your PC (Windows, Git Bash, in the repository folder)

1. Download the folder `project17_asr/export/cynipsnet_inen1` from Google Drive, for example to
   `C:\Users\choprasu\Downloads\cynipsnet_inen1`.
2. Compile it for the NPU and put it into the firmware (same ST Edge AI command and memory pools as
   the current model; it also points `models.json` at the new weights):
   ```bash
   sh model-tools/asr_finetune/st_compile_asr.sh /c/Users/choprasu/Downloads/cynipsnet_inen1
   python tools/models_layout.py          # must print no PROBLEM line
   ```
3. Build and flash (BOOT1 = H, then L and reset):
   ```bash
   cd firmware-n6
   cmake --preset default -G Ninja -B build_translator -DCMAKE_BUILD_TYPE=Release -DSTM32_C_OPTIMIZATION=-Os \
       -DASR_MODEL_NETWORK=cynipsnet_inen1 -DSTEDGEAI_ROOT=C:/ST/STEdgeAI/4.0/Middlewares/ST/AI -DAPP_TRANSLATOR_MODE=ON
   cmake --build build_translator --target app
   cd ..
   python tools/flash.py n6 --models
   ```
   To go back to the old recognizer: `git checkout firmware-n6/src/application/translator/models.json`,
   build with `-DASR_MODEL_NETWORK=cynipsnet_punct1`, flash the same way.

**Send back:** the table printed by cell 12, the output of `st_compile_asr.sh` (the epochs line must
say all are pure hardware, and the input scale must be 0.0628561676), and a few sentences you said
to the board with what it wrote.
"""


def lines(src):
    src = src.strip("\n") + "\n"
    out = src.splitlines(keepends=True)
    out[-1] = out[-1].rstrip("\n")
    return out


def main():
    with open(os.path.join(HERE, "asr_lib.py")) as f:
        lib = f.read()
    code = [SETUP, None, SETTINGS, CHECK_MODEL, DOWNLOAD, INDEX, DEV, BASELINE, TRAIN, RESULTS, EXPORT, BOARD]
    cells = [{"cell_type": "markdown", "metadata": {}, "source": lines(INTRO)}]
    for c in code:
        if c is None:
            src = "%%writefile /content/asr_lib.py\n# 2. The training code (model-tools/asr_finetune/asr_lib.py)\n" + lib
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                          "outputs": [], "source": lines(src)})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": lines(c)})
    cells.append({"cell_type": "markdown", "metadata": {}, "source": lines(PC_STEPS)})
    nb = {"nbformat": 4, "nbformat_minor": 0,
          "metadata": {"colab": {"provenance": [], "gpuType": "T4", "name": "asr_indian_english.ipynb"},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "language_info": {"name": "python"}, "accelerator": "GPU"},
          "cells": cells}
    out = os.path.join(HERE, "asr_indian_english.ipynb")
    with open(out, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("wrote", out, "(%d cells)" % len(cells))
    if "--script" in sys.argv:
        script = os.path.join(HERE, "notebook_cells.py")
        with open(script, "w") as f:
            f.write("# generated by make_notebook.py --script: the notebook's code cells, for a PC test\n")
            for c in code:
                if c is not None:
                    f.write(c.strip("\n") + "\n\n")
        print("wrote", script)


if __name__ == "__main__":
    main()
