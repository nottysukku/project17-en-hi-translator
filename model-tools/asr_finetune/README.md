# Fine-tuning the DK's speech recognizer for Indian-accented English

`asr_indian_english.ipynb` runs in Google Colab (T4 GPU). It takes the recognizer the board runs
now (CynipsNet with punctuation, `cynipsnet_punct1`), trains it further on English spoken with
Indian accents, and writes an int8 ONNX file for ST Edge AI with the same layers, size and memory
placement. The new model drops into the same flash slot (0x71500000, 7.0 MB) and RAM pools.

## What the owner does

1. Copy `cynipsnet_punct1.onnx` (from `cynipsnet_re\deploy\out\cynipsnet_punct1\` on the PC) to
   Google Drive, folder `project17_asr`.
2. Open the notebook in Colab (File > Upload notebook, or from GitHub), choose a T4 GPU,
   Runtime > Run all. Optional: a Hugging Face token as Colab secret `HF_TOKEN`, for the Svarah test.
3. When a session ends, Run all again: it continues from the last checkpoint on Drive.
4. When the results cell says training finished, the last cells export
   `project17_asr/export/cynipsnet_inen1/` and the final text cell gives the PC commands
   (`st_compile_asr.sh`, build, flash).

## Facts this rests on

- The "268 epochs" in `asr_engine/models/cynipsnet_v1.0/README.md` ("268 of 268 epochs are pure
  hardware") are NPU scheduling steps, not training passes. The original fine-tuned model was
  compiled from `epoch_04_step_00021311_v9.tflite` (converter log): its fine-tuning stopped at
  training epoch 4. Fine-tuning here starts from that model; 16 epochs over ~460 h is the default.
- Features: `asr_lib.LogMel` and `asr_lib.board_windows` equal the board's `mel.c` and `norm.c`.
  Checked with `../asr_feature_check.py` against `firmware-n6/src/asr_engine/test/feature_host.c`
  on Common Voice accented speech: log-mel within 2e-4, all 5,030,400 int8 window codes identical.
- The network is read from the ONNX file the board's model was compiled from (`OnnxNet`), trained,
  and written back into the same graph; the int8 export keeps the board's input scale
  (0.0628561676, zero point 0) and adds an int8 output with its scale in `network_shape.h`.
- That file (input `[1, 64, 1, 600]`, 2-D convolutions) has no Relu nodes: each ReLU is an int8
  step whose zero point is the lowest code (-128), so it cuts negatives off. `OnnxNet` turns such
  steps into ReLUs (CynipsNet has 33) and the float export writes them as Relu nodes; the int8
  export quantizes activations asymmetrically so they fold back the same way. Version 1 of the
  notebook skipped these steps: the old model scored 100% and the losses were `nan`.
- The board simulation (`BoardSim`) feeds each check set as one continuous stream (the board's
  statistics cover the last 30 s); its window mechanics were checked against whole-utterance
  decoding with a model whose context fits the window (all steps away from the edges equal).
  Training blends some rows' statistics with the batch's (`norm_mix`) so the model does not
  depend on per-sentence normalization.

## Data (Hugging Face, downloaded by the notebook, about 47 GB)

| Source | Use | Hours | Transcripts |
|---|---|---|---|
| `ishands/commonvoice-indian_accent` (CC0) | train; 2% of speakers held out as a check set | 164 | punctuated |
| `skbose/indian-english-nptel-v0` | train (40 of 215 files); test split as a check set | ~170 | upper case, no marks |
| `SPRINGLab/IndicTTS-English` | train (4 of 28 files) | ~30 | no marks used |
| `openslr/librispeech_asr` + LibriSpeech-PC (openslr.org/145) | replay, 100 h per epoch; dev-clean as a check set | 180 pool | punctuated (91%) |
| `ai4bharat/Svarah` (CC BY 4.0, gated) | final test only | 9.6 | - |

Training rules: transcripts without punctuation hide the `, . ?` classes from the loss (the model
is not taught to drop marks); LibriSpeech rows also stay close to the old model's outputs (a KL
term, `kl_weight`), so what the board learned before (its microphone, noise) is not forgotten.
Augmentation: speed 0.9/1.1, room echo, band limits, noise or babble at 3-30 dB, levels from -26
to -3 dBFS, quiet lead and tail.

## Files

- `asr_lib.py`: everything (features, ONNX graph in PyTorch, data, training, export, board
  simulation). The notebook writes it to `/content` in its second cell.
- `make_notebook.py`: builds the notebook from `asr_lib.py` (`--script`: its code cells as one
  script, for a PC test with `ASR_QUICK_TEST=1`).
- `smoke_test.py`: PC test with a stand-in network of the same layers: load from a QDQ ONNX file
  stored like the owner's (and two other layouts; all 33 ReLUs found), train (loss falls), loader, checkpoints, float ONNX back out (1e-6 from PyTorch),
  int8 with the board's input scale, board simulation, the training loop with two workers.
- `st_compile_asr.sh`: on the owner's PC, ST Edge AI with the same command as `cynipsnet_punct1`,
  then the files into `firmware-n6/src/asr_engine/models/<name>/` and `models.json`.

PC checks (from this folder, with the venv of HANDOFF.md section 8):

```bash
python smoke_test.py /tmp/asr_smoke <a parquet file with "audio" and "sentence" columns>
ASR_QUICK_TEST=1 ASR_DRIVE=/tmp/nb/drive ASR_WORK=/tmp/nb/work ASR_LIB_DIR=$PWD \
  ASR_START_ONNX=/tmp/asr_smoke/standin_cf_qdq.onnx python make_notebook.py --script && python notebook_cells.py
```

The second runs every notebook cell on one file of each real dataset (about 3 GB) on the CPU.
