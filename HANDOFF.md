# Handoff: state of Project 17 and what to do next

Written 2026-10-22 by the Claude Code session that built this, for the agent continuing it. Read this whole file before changing anything.

## 1. Goal (the owner's words, condensed)

- The person speaks English into the DK's microphone. It is converted to Hindi "PERFECTLY": any model may be used, and long Colab training is acceptable.
- The Hindi is shown in Devanagari and spoken with a neural voice.
- An Android phone (with LE Audio) streams music to the WBA. Music has top priority: while it plays, no translation happens. When it stops, translation resumes by default.
- There is a good touch GUI, with manual switching.
- The N6 must sleep while the WBA music is active.
- Use: an internal demo / education.
- The owner wants all firmware and model work done. Their part is to connect the boards, say "flash", have it flashed, and test.

## 2. Hardware (audio output & wiring)

- **STM32N6570-DK:**
  - Cortex-M55 at 800 MHz with Helium, plus the Neural-ART NPU.
  - 32 MB PSRAM at 0x90000000.
  - 128 MB octal NOR at 0x70000000.
  - 800x480 touch LCD.
  - Onboard 3.5 mm stereo audio jack (TRRS) driven by WM8904 codec.
- **NUCLEO-WBA55CG:** BLE only, LE Audio music receiver.
- **Audio output specification for translated Hindi (on-the-fly speech):**
  - **Output port:** Onboard 3.5 mm audio jack on the STM32N6570-DK.
  - **Amplifier:** PAM8403 Class-D stereo 3 W amplifier module.
  - **Power & Filtering:** 5 V (DK CN8-5) and GND (DK CN8-6). A **470 µF electrolytic capacitor** must be placed directly across VCC and GND close to the PAM8403 board to absorb current spikes and prevent brownouts / audio pops.
  - **Speaker:** 4 Ω, 3 W speaker connected across the PAM8403 speaker output terminals (L+ / L- or R+ / R-). *Note: PAM8403 outputs are bridge-tied loads (BTL); neither speaker terminal must ever be connected to ground.*
- **Alternate / Shared Signal Path (I2S & Music Receiver):**
  - TMUX1574 2:1 switch → MAX98357A (GAIN = 100 kΩ to VIN, 3 dB) → speaker, with WBA SAI1_B on side A and N6 SAI2_B on side B.
- **Control Pins:**
  - N6 I2S: SAI2_B on A2/D6/D8.
  - WBA I2S: SAI1_B on PB6/PB5/PB7.
  - TMUX SEL: N6 D7 = PD6.
  - Amp SD: N6 D4 = PH5.
  - Music line: WBA PB0 → N6 D2 = PD0.
- Full table: `firmware-n6/src/application/translator/README.md`.

## 3. What exists

### firmware-n6 (branch history included)

- **ASR:** CynipsNet CTC on the NPU, fine-tuned for Indian English, with punctuation and sentence case. The model is in `src/asr_engine/models/cynipsnet_punct1` (weights flashed at 0x71500000).
- **Translator mode** (`src/application/translator/`): built and flashed once; not yet tested with the wiring.
  - `xl_audio_out.c`: SAI2_B I2S at 48 kHz, with a −3 dB level ceiling sized for the 0.5 W speaker.
  - `xl_router.c`: SEL, amp mute, and the music-line debounce (on after 100 ms, off after 1.5 s).
  - `xl_display.c`: GUI with status bar, English panel, and a Hindi placeholder. Buttons are AUTO / TRANSLATE / MUSIC / TONE.
  - **"Sleep":** recognition is paused (`audio_port_set_paused`), audio out stops, the LCD turns off and the CPU waits in WFI.
- **`src/translate/it2.c`:** IndicTrans2 English→Hindi engine in portable C (Helium `q4_group_dot` under `__ARM_FEATURE_MVE`). Tested on the PC only; not yet in any CMake target.
  - Blob made by `model-tools/it2_pack.py`: 18 encoder + 18 decoder layers, 4-bit weights in groups of 64 with fp16 scales, and a Hindi-only output vocabulary of 41052 tokens (`model-tools/assets/hindi_vocab.json`). It is 93.0 MB.
  - Activations are int8 with one scale per 64-group. Per-row scales cost 6 chrF.
  - **Measured on 120 IITB test sentences:** chrF++ 47.7, against 48.1 for the float model. On the PC it takes 205 ms to encode and 290 ms to decode per sentence. Board speed is unknown: the decoder reads about 44 MB of weights per token from NOR.
  - **Tokeniser:** matches Python on 133 of 136 sentences. Only abbreviations such as "Dr." differ; adding a non-breaking-prefix list would fix that.
  - **PC test:**
    - Build: `python -m ziglang cc -O2 -I. it2.c test/it2_host.c -lm`.
    - Run: `model-tools/it2_host_check.py`.
- **`src/application/translator/xl_hindi_text.c`:** Devanagari rendering from `model-tools/assets/hi_font_40.bin`. Not yet called by the GUI.
  - The font has 6722 syllables pre-shaped with HarfBuzz from Noto Sans Devanagari (OFL), 40 px, weight 500, 5.2 MB.
  - Syllable splitting is identical to Python on 20000 sentences, and wrapping works. PC test: `test/hindi_text_host.c`.
  - `mw_lcd_draw_alpha4()` was added to `src/middleware/lcd`.

### firmware-wba

- ST `BLE_Audio_TMAP_Peripheral` (STM32CubeWBA 4ccfd152), ported to the Nucleo. All changes are under `BOARD_NUCLEO_MAX98357A`.
- Builds with CMake against an STM32CubeWBA checkout with submodules. It has **not been flashed or tested**.
- Details are in its README:
  - software volume plus a −3 dB ceiling;
  - PB0 music line (audible stream, held 1 s);
  - advertises as `N6_Music_xxxx`;
  - media receiver roles only.

### Text-to-speech (model-tools; nothing in the firmware yet)

- **Model:** `facebook/mms-tts-hin` (VITS, character input, 16 kHz, CC-BY-NC, which is fine for an internal demo).
- **Components:**
  - text encoder: 6.3 M parameters;
  - stochastic duration predictor (SDP): 1.3 M;
  - flow: 7.1 M;
  - HiFi-GAN: 14.3 M.
  - The posterior encoder is not needed.
- **Plan:**
  - text encoder + SDP + length regulation + noise on the CPU in C (not written yet);
  - flow and HiFi-GAN on the NPU (graphs done, see below);
  - resample 16 → 48 kHz for `xl_audio_out`;
  - pause ASR while speaking.
- **NPU findings (ST Edge AI 4.0, measured):**
  - Vertical kernels are limited to 3.
  - Conv rows wider than about 768 fall back to software.
  - ConvTranspose becomes a software Resize.
  - A dilated kernel spanning more than about 11 runs in software.
  - Div and some Adds stay float.
- **`tiled_vocoder.py`:** HiFi-GAN rewritten as an exact static graph (max error 3e-7 against the original). It has:
  - time on the width axis;
  - a 3-tap polyphase conv instead of ConvTranspose;
  - rows of at most 640 with halos, re-tiled per stage;
  - valid convolutions;
  - big dilated kernels split into 3-tap groups;
  - 1/3 averaging folded into the weights.
  - **Compiled:** everything on the NPU except the input quantize, output dequantize and tanh. That is 24.6 GMAC per 64-frame chunk (1.024 s of audio), 16 MB of int8 weights, 12.7 MB of activations in hyperRAM. The board needs 15 latent frames of context on each side.
- **`npu_flow.py`:** the flow rewritten exactly (1.4e-6), with flips folded into the weights and a 512-frame static length plus mask. It has **not been compiled yet**.
- **int8 quality is not verified yet.** `tts_eval.py` runs float against int8 (ONNX Runtime) and scores with Whisper-small; it was interrupted. The vocoder's waveform SNR was about 15 dB. If the voice sounds bad, try better calibration (percentile/entropy, more data) before anything else.

## 4. What the cloud agent can and cannot do

- **Can:** write and refactor C and Python, build and run PC tests (gcc/clang instead of zig on Linux), cross-compile with `arm-none-eabi-gcc` if it installs, and download Hugging Face models if the network allows.
  - Use the ungated mirror `naklitechie/indictrans2-en-indic-dist-200M`; its checksums match the gated original.
  - `model-tools/fetch_models.py` downloads it.
- **Cannot:** flash or test hardware, run ST Edge AI (proprietary, Windows install at `C:\ST\STEdgeAI\4.0`), or run Windows paths. `st_compile.sh` and `flash.py` contain the owner's local paths.
- **Hand the owner exact commands** for anything that needs the ST tools. The NPU graphs must be compiled on their PC:
  1. `model-tools/tts_export.py ... 64 t` exports and quantizes the tiled vocoder.
  2. `st_compile.sh` compiles it. Give each network its own `stedgeai --name` and an mpool whose octoFlash offset matches its final flash address.
- **firmware-n6 builds** against the ST Edge AI 4.0 runtime (`-DSTEDGEAI_ROOT=C:/ST/STEdgeAI/4.0/Middlewares/ST/AI`), because `cynipsnet_punct1` was generated by 4.0. The vendored runtime under `src/middleware/stedgeai` may not match.
- **Large generated files are not in git:**
  - the 93 MB MT blob: regenerate with `fetch_models.py` + `it2_pack.py --dec-layers 0,1,...,17`;
  - the ONNX graphs;
  - the IITB data (`cfilt/iitb-english-hindi`, ungated).

## 5. Next tasks, in order

1. **NOR flash budget (128 MB).** It currently over-runs:
   - FSBL 1 + app ≤4 + ASR 7 (fixed at 0x71500000) + MT 93 + vocoder 16 + flow ~7 + TTS CPU weights ~4 + font 5.2 ≈ 137 MB.
   - Free regions are 0x70500000–0x71500000 (16 MB) and 0x71C00000–0x78000000 (100 MB).
   - Options:
     - trim the MT source vocabulary to the most frequent ~20k BPE pieces (saves about 5 MB);
     - trim the target vocabulary to 31k tokens (99.85% coverage, about 2.5 MB);
     - drop 1–2 decoder layers, but measure chrF first;
     - use a 36 px font.
   - Write `firmware-n6/src/application/translator/models.json` with each blob and its address; `flash.py --models` reads it.
2. **TTS quality check** with `tts_eval.py`, then fix the quantization if needed.
3. **TTS front end in C.** Verify each piece against PyTorch (`transformers` `VitsModel`) on the PC:
   - MMS tokenizer: characters with `add_blank`, vocabulary in the model's `vocab.json`;
   - text encoder with relative attention (window 4);
   - SDP reverse with rational-quadratic spline flows;
   - durations: `ceil(exp(logw))`;
   - prior noise (scale 0.667, duration noise 0.8).
4. **Firmware integration** in translator mode:
   - Link `it2.c` and `xl_hindi_text.c`.
   - Add a translator thread with lower priority than ASR. Use the PSRAM workspace (`it2_workspace_size`, about 20 MB with 18 decoder layers; consider fp16 KV caches).
   - Detect the end of a sentence from the ASR text (end mark, or a pause) and translate each sentence.
   - Show the Hindi panel.
   - Run TTS with ASR paused while speaking.
   - Stop speech immediately when music starts (router).
   - Add the two extra NPU networks.
5. **Speed on the board:** measure, then stream weights with DMA if NOR reads by the CPU are slow. Encoder GEMMs already reuse each weight row for all tokens.
6. **Optional:** fine-tune or distil a shallower decoder for speed (Colab). With 10 of 18 layers and no fine-tuning, chrF drops to 43.1 at 4-bit.
7. **True Stop mode** on the N6. Today the "sleep" is Sleep mode: WFI, NPU idle, LCD off.

## 6. Conventions

- **Commits:** end messages with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Work on a branch, and do not push to `main` without the owner.
- **Code style:** same as the existing files. Doxygen file and function headers, short plain comments, C99, no dynamic allocation in firmware.
- **The owner:** works on Windows and is not a full-time embedded developer.
  - Give short, plain status updates: the outcome first, then what they need to do.
  - When they need to act on the PC, give exact commands.
  - The N6 is flashed with BOOT1 = H, then set back to L and reset.
