# Translator mode

Build with `-DAPP_TRANSLATOR_MODE=ON`. In this mode the STM32N6570-DK recognizes English and shows
the Hindi translation in Devanagari script.

## Wiring

One speaker plays both the Hindi voice and the WBA's Bluetooth music:

- **Output:** the DK's own **3.5 mm audio jack** (WM8904 codec, headphone output) → a
  **PAM8403** class-D module (gain 24 dB) → a **4 Ω 3 W speaker**. A **470 µF** electrolytic
  (16 V or more) sits straight across the PAM8403's +5 V and GND pins, at the module.
- **Music in:** the NUCLEO-WBA55CG sends its decoded music as I2S, through a **TMUX1574**, to the
  DK (SAI2 block B, slave receiver on A2/D6/D8). The DK plays it on the jack when the music has
  the floor. No MAX98357A.
- **TMUX1574:** a gate between the boards. SEL is tied to GND (side A only). EN is active low,
  driven by DK D7 with a 10 kΩ pull-up: the switch stays open until the firmware has made A2, D6
  and D8 inputs, and its powered-off protection keeps the boards apart while the DK is off.
- Everything runs from the boards' USB cables: no battery or separate supply.

[wiring.html](wiring.html) draws the whole setup with the connections in wiring order; open it in
a browser. The same as images: [wiring_schematic.png](wiring_schematic.png) (the drawing) and
[wiring_full.png](wiring_full.png) (the drawing, the numbered steps and the notes).

![Wiring of the whole setup](wiring_schematic.png)

**Flash the new firmware before wiring the TMUX.** The firmware before commit "Translator audio:
Hindi voice and WBA music both out of the DK's 3.5 mm jack" drives A2, D6 and D8 as outputs.

**Power**

- PAM8403 +5V: DK CN8-5, the board's +5 V output; GND: CN8-6. The 470 µF goes across these two
  pins at the module: long leg / unmarked side to +5V, stripe to GND.
- DK power: USB-C on CN6 (ST-LINK), from a USB-C port or charger. UM3300 says the board then gets
  up to 1.66 or 3.2 A. An old USB-A port can limit it to 0.55 A, too little for loud music.
- Nucleo power: its own ST-LINK USB cable.
- GND: one common ground for the DK (CN8-6), the Nucleo (CN5-6), the TMUX (pin 8) and the PAM8403.
- TMUX1574 VDD (pin 16): DK CN8-4 (+3.3 V), with 100 nF to pin 8 right at the chip.

**TMUX1574PWR (TSSOP-16, on a TSSOP-16 to DIP adapter).** Pin order from TI's data sheet
SCDS391C; pin 1 is at the dot.

| Pin | Name | Connect to |
|---|---|---|
| 1 | SEL | GND (side A always) |
| 2 | S1A | Nucleo PB6 (CN8-4, "D3"): WBA bit clock |
| 3 | S1B | GND (unused) |
| 4 | D1 | DK A2 = PE12 (CN7-3), SAI2_SCK_B |
| 5 | S2A | Nucleo PB5 (morpho CN4-35): WBA word select |
| 6 | S2B | GND (unused) |
| 7 | D2 | DK D6 = PE13 (CN11-7), SAI2_FS_B |
| 8 | GND | ground |
| 9 | D3 | DK D8 = PE7 (CN12-1), SAI2_SD_B |
| 10 | S3B | GND (unused) |
| 11 | S3A | Nucleo PB7 (CN8-3, "D2"): WBA data |
| 12 | D4 | DK D2 = PD0 (CN11-3), with 100 kΩ to GND on the DK side: music line |
| 13 | S4B | GND (unused) |
| 14 | S4A | Nucleo PB0 (CN8-7, "D6"): music line, high while music plays |
| 15 | EN | DK D7 = PD6 (CN11-8), and 10 kΩ to +3.3 V (pin 16) |
| 16 | VDD | +3.3 V (DK CN8-4), 100 nF to pin 8 |

**Jack and PAM8403**

- A 3.5 mm plug in the DK's audio jack: tip → PAM8403 `L`, ring → `R`, sleeve → `G` (input
  ground). Both channels carry the same signal (the music is mixed to mono).
- Speaker: 4 Ω 3 W across `L+` and `L−` only. The outputs are bridge-tied: never connect a speaker
  wire to ground, to the other channel, or to the USB shield. A second 4 Ω speaker may go on
  `R+`/`R−` (twice the current from the DK's 5 V).
- Loudness: the codec's headphone volume is −14 dB (`XL_AUDIO_OUT_VOLUME_DB`), so a full-scale sine
  is about 0.2 Vrms at the jack and about 3.2 Vrms (2.5 W) out of the PAM8403, where it starts to
  clip at 5 V. The Hindi voice is 3 dB lower still (`XL_AUDIO_OUT_CEILING_Q15`); the WBA scales its
  music itself. If the sound distorts or the DK resets on loud music, build with
  `-DXL_AUDIO_OUT_VOLUME_DB=-20`.

Not used, and why:

- PD12 is the DK's Ethernet MDIO line, so EN is on D7.
- PA8 is the Nucleo's ST-LINK VCP RX line, so the WBA uses SAI1 block B.
- The Nucleo user buttons B2 and B3 sit on PB6 and PB7: do not press them while music plays.
- D4 = PH5 (the old amplifier shutdown) is free now.

## Behavior

- **Translation.** The recognized English is cut into sentences (`xl_sentence.c`): at a `.`, `?` or
  `!`, after 1.2 s with no new words, or at 200 characters. Each sentence is translated on the
  board (`src/translate/it2.c`, about 2 to 4 s per sentence, to be measured) and the Hindi appears
  under the last two lines of English. The status bar says "Translating..." meanwhile.
- **Speech.** If the N6 has the speaker, the Hindi is then spoken (`xl_speech.c`): "Speaking
  Hindi..." in the status bar. Numbers in digits are read as Hindi words (`src/tts/tts_numbers.c`). Recognition pauses while it speaks, so it does not hear itself, and
  sentences said meanwhile are not recognized.
- **Speaker.** AUTO mode (the default) gives the speaker to the music while it plays, and to the
  translator otherwise. TRANSLATE and MUSIC force one source. The USER button cycles through the
  three modes.
- **Music always wins.** When music takes the speaker, speech fades out within 5 ms and its queue
  is dropped, a translation in progress is cancelled within one word, and the queued sentences
  and the half-heard one are dropped.
- **Sleep.** While the music has the speaker, the N6 sleeps:
  - recognition is paused, so the NPU is idle;
  - the translator thread waits, idle;
  - the audio stream keeps running: it carries the music from the WBA to the jack (two DMA
    interrupts every 5.3 ms);
  - the screen and backlight go off 5 s later;
  - the CPU waits in WFI.

  A tap or the USER button lights the screen for 10 s. When the music stops (1.5 s without the
  line), or on TRANSLATE, the N6 wakes and beeps twice.
- **TONE** plays 880 Hz on the speaker (jack → PAM8403), to check the wiring.

## Models and memory

`models.json` is the memory map: every model's slot in the 128 MB NOR flash and the PSRAM
regions. `tools/models_layout.py` checks it and writes `xl_models.h`; `tools/flash.py n6 --models`
programs the blobs. Without a blob the firmware still runs: no translation model, a message on the
screen; no font, a message; no voice networks or front-end weights, text only. The log on the
ST-LINK serial port (115200 baud) says which.

| Blob | Made by | Runs on |
|---|---|---|
| Translation (IndicTrans2, 4-bit) | `model-tools/it2_pack.py` | CPU |
| Hindi font | `model-tools/assets/hi_font_40.bin` | CPU |
| Voice front end (MMS-TTS text encoder, durations) | `model-tools/tts_front_pack.py` | CPU |
| Voice flow, vocoder | ST Edge AI from `tts_eval.py` / `tts_export.py` graphs | NPU |
| Speech recognition | `src/asr_engine/models/cynipsnet_punct1` | NPU |

## Test without the WBA

With TMUX pin 12 not connected to D2 (or the TMUX not wired yet), connect a jumper from DK D2
(CN11-3) to +3.3 V (CN8-4). The router sees music playing, gives the jack to the music (silence,
with no WBA) and puts the N6 to sleep. Remove the jumper: about 1.5 s later the N6 wakes and beeps.
With the jumper in while the N6 speaks, the speech must stop at once. Do not use the jumper with
the TMUX wired: it would fight the WBA's PB0 output through the switch.

## PC tests

From this folder (gcc or clang):

```
cc -I. xl_sentence.c test/sentence_host.c -o sentence_host && ./sentence_host
cc -DXL_SPEECH_HOST_TEST -I. -I../../tts xl_speech.c xl_resample.c ../../tts/tts_front.c ../../tts/tts_numbers.c test/speech_host.c -lm -o speech_host
./speech_host ../../../../model-tools/out/tts/tts_front.bin
cc -DXL_HINDI_TEXT_HOST_TEST -I. xl_hindi_text.c test/hindi_text_host.c -o hindi_text_host
cc -I../../tts ../../tts/tts_numbers.c ../../tts/test/numbers_host.c -o numbers_host && ./numbers_host
```

## Not done yet

- Measured speed on the board (translation, voice), and the first test with the wiring (the jack,
  the WBA's music through the TMUX).
- True Stop mode: the microphone driver cannot restart, so "sleep" is currently idle plus WFI.
