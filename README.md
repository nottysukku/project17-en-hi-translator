# Project 17: English to Hindi translator on STM32N6

A person speaks English into the STM32N6570-DK. The board shows the English (on-device ASR), translates it to Hindi on the device, shows the Hindi in Devanagari, and speaks it through a MAX98357A amplifier and an 8 Ω 0.5 W speaker.

A NUCLEO-WBA55CG receives Bluetooth LE Audio music from an Android phone and plays it through the same speaker, via a TMUX1574 switch. Music has priority: while it plays, the N6 stops translating and sleeps. When the music stops, translation resumes. A touch-screen GUI on the DK allows switching by hand.

Everything runs offline on the two boards.

| Folder | What it is |
|---|---|
| `firmware-n6/` | STM32N6570-DK firmware (CMake, ThreadX). Translator mode: `-DAPP_TRANSLATOR_MODE=ON`. Started from fotonik-sk/stm32n6-speech-recognition (Apache-2.0); full history kept. |
| `firmware-wba/` | NUCLEO-WBA55CG LE Audio music receiver: ST's TMAP example, ported. See its README. |
| `model-tools/` | Python tools that build the on-device models: IndicTrans2 packing, the Devanagari syllable font, and the TTS graphs for the NPU. |
| `tools/flash.py` | Flash both boards; each probe is picked by board name. |

Wiring: `firmware-n6/src/application/translator/README.md` and `wiring.html` / `wiring_schematic.png` in the same folder.

**Status and next steps: see [HANDOFF.md](HANDOFF.md).**
