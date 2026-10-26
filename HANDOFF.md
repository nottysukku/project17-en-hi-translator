# Handoff: state of Project 17 and what to do next

First written 2026-10-22 by the Claude Code session that built the models; updated the same day by the
cloud session that did tasks 1-4 of the old list (branch `claude/peaceful-mccarthy-gdlqoe`). Read this whole
file before changing anything.

## 1. Goal (the owner's words, condensed)

- The person speaks English into the DK's microphone. It is converted to Hindi "PERFECTLY": any model may be used, and long Colab training is acceptable.
- The Hindi is shown in Devanagari and spoken with a neural voice.
- An Android phone (with LE Audio) streams music to the WBA. Music has top priority: while it plays, no translation happens. When it stops, translation resumes by default.
- There is a good touch GUI, with manual switching.
- The N6 must sleep while the WBA music is active.
- Use: an internal demo / education.
- The owner wants all firmware and model work done. Their part is to connect the boards, say "flash", have it flashed, and test.

## 2. Hardware (as wired now, 2026-10-25)

- **STM32N6570-DK:**
  - Cortex-M55 at 800 MHz with Helium, plus the Neural-ART NPU.
  - 32 MB PSRAM at 0x90000000.
  - 128 MB octal NOR at 0x70000000.
  - 800x480 touch LCD.
  - WM8904 codec (SAI1 block A on PG7/PB6/PB7/PB0, I2C2 address 0x34, shared with the touch
    controller) with a 3.5 mm audio jack.
- **NUCLEO-WBA55CG:** BLE only, LE Audio music receiver; I2S out on SAI1 block B (PB6/PB5/PB7).
- **Speaker (the owner's choice):** DK 3.5 mm jack → PAM8403 module (470 µF across its 5 V input,
  at the module; 5 V from DK CN8-5) → 4 Ω 3 W speaker on L+/L−. **No MAX98357A.**
- **Music path:** WBA I2S → TMUX1574 side A → DK SAI2 block B (slave receiver on A2/D6/D8) → the
  DK plays it on the jack. TMUX SEL = GND; EN = DK D7 (PD6), active low, 10 kΩ pull-up: open until
  the firmware has made A2/D6/D8 inputs. Music line: WBA PB0 → TMUX S4A/D4 → DK D2 (PD0), 100 kΩ to
  GND. D4 (PH5) is no longer used.
- Full table, drawing and wiring order: `firmware-n6/src/application/translator/README.md`,
  `wiring.html`, `wiring_schematic.png`.

## 3. Status in one paragraph

Everything is written and tested on the PC; nothing new has run on the board yet. The firmware's translator
mode now cuts the recognized English into sentences, translates each on the board, shows the Hindi, and speaks
it, with music cancelling translation and speech at once. All models fit the 128 MB NOR (map in
`firmware-n6/src/application/translator/models.json`). What is left for the owner: compile the two voice
networks with ST Edge AI, build, flash, and send back the logs (section 6). The next agent then fixes what the
board shows and measures speed (section 7).

Added 2026-10-25 (same branch): the audio now leaves through the DK's 3.5 mm jack (WM8904), which also plays
the WBA's music received over I2S through the TMUX1574 (section 2); and a Colab notebook that fine-tunes the
speech recognizer for Indian-accented English (`model-tools/asr_finetune/`). Neither has run on the board yet.

## 4. What exists

### firmware-n6

- **ASR:** CynipsNet CTC on the NPU, Indian English, punctuation and sentence case
  (`src/asr_engine/models/cynipsnet_punct1`, weights at 0x71500000). It rarely writes `.` (recall 3%) or `?`,
  so sentences are mostly cut at pauses.
  - `asr_engine_npu_acquire()/release()` lets the translator borrow the NPU: it holds the recognizer's input
    semaphore, so no window runs or is written while the voice networks use the shared NPU memories.
- **Translator mode** (`src/application/translator/`, `-DAPP_TRANSLATOR_MODE=ON`), see its README:
  - `xl_audio_out.c`: SAI1 block A I2S master + MCLK into the WM8904 (ST's `stm32-wm8904` V1.1.1 in
    `src/hardware/components/wm8904`), headphone volume -14 dB; music in on SAI2 block B (slave, GPDMA1 ch. 5),
    mixed to mono and read at the jack's rate by a PI loop on the ring level (simulated: stable 16-48 kHz and
    with 1 % clock mismatch). Source changes fade over 5 ms in the DMA interrupt. `xl_router.c`: AUTO /
    TRANSLATE / MUSIC choose the source; D7 opens and closes the TMUX.
  - `xl_sentence.c`: sentence cutting (end mark not after Mr./Dr./...; 1.2 s pause; 200 characters).
  - `xl_translator.c`: the translator thread (priority 13, lowest). Per sentence: sentence case, `it2_translate`,
    publish the Hindi, then (if the N6 has the speaker) speak with recognition paused and the NPU borrowed.
    `xl_translator_cancel()` stops it; `xl_translator_listen()` combines the pause reasons (music, speech).
  - `xl_speech.c`: voice front end (CPU) → flow (NPU, 512-frame windows, 32 frames of context) → vocoder (NPU,
    64 frames + 15 each side) → `xl_resample.c` (16 → 48 kHz, 96 taps) → speaker ring; audio is queued once two
    vocoder steps are ahead. Logs the time of each stage ("Speech: ...").
  - `xl_display.c`: two lines of English, the Hindi below (`xl_hindi_text.c`), "Translating..." / "Speaking
    Hindi..." in the status bar. On music: flush the speech (the stream keeps running: it carries the music), cancel, drop the half-heard sentence.
  - `models.json` → `tools/models_layout.py` → `xl_models.h` (addresses). The linker's NOR_APP now ends at
    0x70500000 (4 MB; the app is 1.24 MB without the voice networks' generated code).
  - Not yet tested with the wiring, and the speed is unknown.
- **`src/translate/it2.c`** (IndicTrans2): now in the build (`-O2` always). New: tokeniser splits trimmed source
  pieces like SentencePiece (unused here, see 5), 8-bit scale codes (tensor kind 2, blob version 3), a cancel
  hook, and a fix: `scratch_of()` ignored the 32-byte rounding of the decoder table, which would have crashed
  on the 32-bit board with 18 decoder layers.
- **`src/tts/tts_front.c`**: MMS-TTS front end (tokenizer, text encoder with relative attention, stochastic
  duration predictor with spline flows, durations, prior noise from a counter-based generator). Int8 text
  encoder with int8 activations (as it2), float16 duration predictor.
- **`src/tts/tts_numbers.c`**: numbers in digits become Hindi words before speech (the voice's alphabet has
  only 0-4 and 8, and the translator writes e.g. "25 साल"): Indian system (हज़ार, लाख, करोड़), decimals,
  percent; more than 9 digits are read one by one.
- **`src/tts/xl_voice_net.c`**: the flow and vocoder through ST's generated API (`stai_flow_*`,
  `stai_vocoder_*`). Static assertions check shapes and float32 I/O. CMake builds them in when the generated
  files are in `src/tts/networks/flow` and `src/tts/networks/vocoder`; otherwise text only.
- **`xl_hindi_text.c`** reads font version 2 (run-length coded, decoded one row at a time).

### firmware-wba

Unchanged: ST `BLE_Audio_TMAP_Peripheral` ported to the Nucleo, built with CMake, **not yet flashed or tested**.
Details in its README.

### model-tools

- Speech recognition, Indian English: `asr_finetune/` (Colab notebook `asr_indian_english.ipynb`, built from
  `asr_lib.py` by `make_notebook.py`; `smoke_test.py`; `st_compile_asr.sh`; README). It starts from the ONNX
  file `cynipsnet_punct1` was compiled from (on the owner's PC: `cynipsnet_re\deploy\out\cynipsnet_punct1\`),
  keeps every layer, and exports int8 with the board's input scale. Features checked against the board:
  `asr_feature_check.py` + `firmware-n6/src/asr_engine/test/feature_host.c` (all int8 codes equal). The "268
  epochs" in the CynipsNet README are NPU scheduling steps; the original fine-tune stopped at training epoch 4
  (`epoch_04_step_00021311_v9.tflite`).

- Translation: `it2_pack.py` (`--scale8`, `--src-keep`), `vocab_trim.py --max N`, `it2_trim_eval.py` (packs and
  scores variants in parallel), `it2_host_check.py` (`--save`, `--compare`), `assets/hindi_vocab_31k.json`.
- Font: `hi_font_rle.py` (v1 → v2 converter); `hi_font_build.py` writes v2; `assets/hi_font_40.bin` is v2.
- Voice: `tts_front_ref.py` (Python mirror of the C front end), `tts_front_pack.py`, `tts_front_check.py`,
  `tts_front_eval.py`; `tts_eval.py` now also reports a log-mel distance; `resample_design.py`.
- ST Edge AI: `mpool_edit.py` (pool file for a voice network from `models.json`); `st_compile.sh` takes the
  network name.

## 5. Measured results

**Translation** (C engine, 400 IITB test sentences, chrF++; "same" = sentences translated as the full model):

| Variant | Size | chrF++ |
|---|---|---|
| Full: 18 decoder layers, 41k Hindi tokens, fp16 scales | 93.0 MB | 49.7 |
| 31k Hindi tokens (99.83% coverage; 99% same) | 90.1 MB | 49.6 |
| 31k + 8-bit scales | 87.5 MB | 49.6 |
| 31k, decoder layer 1 / 5 / 9 / 13 / 16 dropped | 87.8 MB | 49.5 / 49.6 / 49.5 / 49.5 / 49.2 |
| **31k + 8-bit scales + layer 5 dropped (used)** | **85.3 MB** | **49.5** |
| Source vocabulary 24k / 20k / 16k / 12k rows (not used) | 90.7-87.5 MB | 49.3 / 48.9 / 48.5 / 47.9 |

On the old 120-sentence set the used model scores 47.9, the same as the full model on this PC (the owner's PC
measured 47.7 for the full model). The C tokeniser matches the Python one on all 136 test sentences (it now
keeps Moses' non-breaking prefixes, "Mr." "Dr." "Rs."), and on 18159 of 18855 IITB training sentences; the rest
have quotes, brackets, "...", "%" or currency, which the recognizer does not write.

**Voice:**
- Front end precision (Whisper-small CER, 12 sentences, float back end): float 16.5%, int8 17.9%, 4-bit 22.2%.
- C front end vs Python: identical structure (with float activations: within 2.7e-5, all 854 durations equal).
- Int8 flow + vocoder in ONNX Runtime vs float (`tts_eval.py`): CER 19.4% vs 17.9%; log-mel distance 0.17 dB,
  where two float renderings with different noise differ by 0.99 dB. The calibration stays as it is.
- Speech pipeline with stand-in networks: sample-exact across windows and steps; cancel queues nothing.

**Speech recognition, Indian English (owner's Colab run of notebook v2, 2026-10-26):** 16 epochs, 76.6k steps,
8,280 h of audio processed at 400-1300x real time on a T4 (about 11 h). Best checkpoint at step 68,000 (`best.pt`,
exported as `export/cynipsnet_inen1` on the owner's Drive; output scale 0.474794).

| WER | old (`cynipsnet_punct1`) | new (`cynipsnet_inen1`) |
|---|---|---|
| Float, whole sentences, 400 each: CV India held-out speakers / NPTEL test / LibriSpeech dev-clean | 38.3 / 54.8 / 6.7 | 24.8 / 32.1 / 7.9 |
| Board simulation (int8, windows, one stream), 40 each: CV India / NPTEL / LibriSpeech | 44.6 / 51.1 / 4.8 | 36.6 / 29.0 / 5.2 |
| Board simulation, **Svarah** (300, never trained on, 117 Indian speakers) | **29.9** | **29.3** |

CV and NPTEL check sets share sentences or lecturers with their training sets, so their gains overstate; Svarah is
the honest measure and barely moved; LibriSpeech (US English) got slightly worse. Next: the board, with the owner's
voice. If that is not better, options: more accent variety (all 28 IndicTTS files, 13 first languages), a lower
`kl_weight`, or the owner's own recordings (the capture build, `-DAPP_AUDIO_CAPTURE=ON`, has no PC receiver yet).

**Flash map** (bytes used / slot): voice front end 7.74 / 7.80 MB at 0x70500000; 1.1 MB free (app growth);
flow ~7.3 / 7.86 MB at 0x70D80000; ASR 7.34 MB at 0x71500000; translation 85.30 / 85.33 MB at 0x71C10000;
font 2.80 / 2.82 MB at 0x76D70000; vocoder ~15.9 / 16.65 MB at 0x77020000 (to the end).
**PSRAM:** app 5 MB; ASR 1 MB at 0x90500000; from 0x90610000, 27 MB used in turn by the translation workspace
(17.7 MB) or the voice (9.4 MB of buffers, then the networks' activations from 0x90F10000).

## 6. What the owner runs (Windows, Git Bash, from the repository folder)

The cloud session cannot run ST Edge AI or flash. Commands, in order:

```bash
# 0. The branch, and ThreadX (its submodule entry was missing before)
git fetch origin && git checkout claude/peaceful-mccarthy-gdlqoe && git submodule update --init

# 1. Models (into model-tools/downloads, about 3.5 GB; skip what you already have there)
python model-tools/fetch_models.py

# 2. Blobs for the CPU
python model-tools/it2_pack.py model-tools/downloads/naklitechie__indictrans2-en-indic-dist-200M \
    model-tools/assets/hindi_vocab_31k.json model-tools/out/it2.bin \
    --dec-layers 0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17 --scale8
python model-tools/tts_front_pack.py model-tools/downloads/facebook__mms-tts-hin model-tools/out/tts/tts_front.bin
sha256sum model-tools/out/it2.bin model-tools/out/tts/tts_front.bin
#   expected b17189e5...4b888 (85302656 bytes) and 59066596...16278 (7736640 bytes)

# 3. The two voice graphs (int8 ONNX)
python model-tools/tts_export.py model-tools/downloads/facebook__mms-tts-hin model-tools/out/tts 64 t
python model-tools/tts_eval.py model-tools/downloads/facebook__mms-tts-hin model-tools/out/tts none --calib-only

# 4. Their memory-pool files, from the pool the recognizer was compiled with
RT=/c/Users/choprasu/Downloads/N6_AutoSpeech/Reverse_engineering2/Reverse_Engineering/cynipsnet_re/roundtrip
mkdir -p model-tools/out/st
python model-tools/mpool_edit.py "$RT/stm32n6.mpool" vocoder model-tools/out/st/vocoder.mpool
python model-tools/mpool_edit.py "$RT/stm32n6.mpool" flow model-tools/out/st/flow.mpool

# 5. Compile both with ST Edge AI 4.0 (each prints its NPU/software split and sizes)
sh model-tools/st_compile.sh model-tools/out/tts/vocoder_t64_int8.onnx model-tools/out/st/vocoder model-tools/out/st/vocoder.mpool vocoder
sh model-tools/st_compile.sh model-tools/out/tts/flow_512_int8.onnx model-tools/out/st/flow model-tools/out/st/flow.mpool flow

# 6. Their C code into the firmware
mkdir -p firmware-n6/src/tts/networks/vocoder firmware-n6/src/tts/networks/flow
cp model-tools/out/st/vocoder/st_out/*.[ch] firmware-n6/src/tts/networks/vocoder/
cp model-tools/out/st/flow/st_out/*.[ch] firmware-n6/src/tts/networks/flow/
python tools/models_layout.py          # must print no PROBLEM line

# 7. Build (the FSBL as before if firmware-n6/fsbl/build is not there yet)
cd firmware-n6
cmake --preset default -G Ninja -B build_translator -DCMAKE_BUILD_TYPE=Release -DSTM32_C_OPTIMIZATION=-Os \
    -DASR_MODEL_NETWORK=cynipsnet_punct1 -DSTEDGEAI_ROOT=C:/ST/STEdgeAI/4.0/Middlewares/ST/AI -DAPP_TRANSLATOR_MODE=ON
cmake --build build_translator --target app
cd ..

# 8. Flash everything (BOOT1 = H, reset; the script waits), then BOOT1 = L and reset
python tools/flash.py n6 --models
```

To try the translation before the voice (steps 3-6 skipped): build as in step 7 (the log then says
"Hindi voice: no networks ... (text only)") and flash with `python tools/flash.py n6 --models --allow-missing`.

**Send back:**
1. The output of steps 5 and 6 (especially lines with `octoFlash`, `hyperRAM`, `HW`/`SW`, and the `.bin`
   names and sizes), and of step 7's configure line "Hindi voice: ..." and the last `NOR_APP` line.
2. The serial log (ST-LINK virtual COM port, 115200 baud) from power-up through two or three spoken sentences:
   lines starting with `Translator:`, `Speech:` and `NPU inference time`.
3. What you see and hear: is the Hindi right, is the voice clear, does it stutter, does music cut it at once.

If a step fails, send its last 30 lines; do not work around it.

## 7. Next tasks, in order

0. **New (2026-10-25):** (a) the owner runs the Colab notebook (`model-tools/asr_finetune/README.md`) and
   sends the table of its last cell; compile with `st_compile_asr.sh`, build with
   `-DASR_MODEL_NETWORK=cynipsnet_inen1`, compare on the board. (b) First test of the jack: log line
   "Audio: WM8904 found, jack at 47991 Hz", TONE, then the WBA's music through the TMUX ("Audio: music input
   at 48000 Hz"). If the WM8904 is not found, the DK may be an older revision with a CS42L51 (not supported).

1. **Get the owner's results through:** fix compile or link errors from the generated code (the file and
   function names `stai_flow_*` / `stai_vocoder_*` and the `.bin` names in `models.json` are assumptions about
   `--name`); check the ST reports: everything on the NPU, the weights within their slots, activations within
   `voice_npu`.
2. **Speed on the board:** from the log. Translation reads ~41 MB of weights per output token from NOR; if
   decoding is slow, stream weights with DMA or keep hot tensors in PSRAM. If the vocoder is slower than real
   time, raise `XL_SPEECH_LEAD_RUNS` or synthesize the whole sentence before playing.
3. **Optional:** distil a shallower decoder for speed (Colab). With 10 of 18 layers and no fine-tuning chrF drops
   to 43.1.
4. **True Stop mode** on the N6. Today the "sleep" is Sleep mode: WFI, NPU idle, LCD off.
5. **WBA:** flash and test the music receiver.

## 8. What a cloud agent can and cannot do

- **Can:** write C and Python; build and run PC tests (gcc); cross-compile the firmware with `arm-none-eabi-gcc`
  (`apt-get install gcc-arm-none-eabi libnewlib-arm-none-eabi`); download from Hugging Face (works); run the
  PyTorch, ONNX Runtime and Whisper checks (a venv with torch CPU, transformers, onnxruntime, sentencepiece,
  sacremoses, sacrebleu, scipy, soundfile). The full model downloads and ONNX exports take about 4 GB.
- **Cannot:** flash, test hardware, or run ST Edge AI.
- The vendored runtime under `src/middleware/stedgeai` is LL_ATON 1.1.3-262; `cynipsnet_punct1` needs -275, so
  cloud builds use `-DASR_MODEL_NETWORK=cynipsnet_v1.0` to check compiling and linking.
- PC tests: `firmware-n6/src/application/translator/README.md` (sentence, speech, Hindi text),
  `model-tools/it2_host_check.py` (build `src/translate` with `gcc -O2 -I. it2.c test/it2_host.c -lm`),
  `model-tools/tts_front_check.py` (build `src/tts` with `gcc -O2 -I. tts_front.c test/tts_front_host.c -lm`).

## 9. Conventions

- **Commits:** end messages with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Work on a branch, and do not push to `main` without the owner.
- **Code style:** same as the existing files. Doxygen file and function headers, short plain comments, C99, no dynamic allocation in firmware.
- **The owner:** works on Windows and is not a full-time embedded developer.
  - Give short, plain status updates: the outcome first, then what they need to do.
  - When they need to act on the PC, give exact commands.
  - The N6 is flashed with BOOT1 = H, then set back to L and reset.
