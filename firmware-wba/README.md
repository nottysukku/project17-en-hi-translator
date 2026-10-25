# NUCLEO-WBA55CG music receiver (Project 17)

A phone with Bluetooth LE Audio streams music to the NUCLEO-WBA55CG. The Nucleo decodes it (LC3)
and sends it as I2S, through side A of a TMUX1574 switch, to the STM32N6570-DK, which plays it on
its 3.5 mm jack (PAM8403, 4 Ω 3 W speaker) instead of the translator's voice. While music plays it
holds a line high, so the translator hands over the speaker and sleeps. Wiring:
`firmware-n6/src/application/translator/README.md`.

The I2S output was written for a MAX98357A, which is no longer used; the signals are the same for
the DK's receiver, so nothing here changed (the names `BOARD_NUCLEO_MAX98357A` and the volume note
below still say MAX98357A).

This is ST's `BLE_Audio_TMAP_Peripheral` example for the STM32WBA55G-DK1 (STM32CubeWBA, commit
4ccfd152), ported to the Nucleo. The first commit is ST's code unchanged, so `git diff` against it
shows the whole port.

## What changed for the Nucleo

All changes are behind `BOARD_NUCLEO_MAX98357A`. Build with `-DBOARD=DK1` to get ST's original.

- **Audio out:** `Board/nucleo_audio.c` replaces the DK1 audio BSP.
  - SAI1 block B, I2S, 16-bit stereo, on the same pins as the DK1: PB6 SCK (Arduino D3), PB5 FS
    (morpho CN4-35) and PB7 SD (Arduino D2).
  - No CS42L51 codec, no I2C and no MCLK.
- **Volume:** the MAX98357A has none, so `Board/music_link.c` scales every decoded sample.
  - The phone's volume (VCP) spans 50 dB.
  - Output is capped at −3 dB, so a full-scale sine is 0.20 W into the 8 Ω 0.5 W speaker, with the
    amplifier's GAIN at 3 dB.
- **Music line:** PB0 (Arduino D6, CN8-7) goes high while a stream plays audible sound. It drops
  within 0.1 s after the stream stops, or after 1 s of silence or volume 0.
- **No screen, joystick or LEDs.**
  - Advertising starts at power-up as `N6_Music_xxxx`, the last four characters being from the
    Bluetooth address.
  - The Nucleo BSP is left out, because its buttons B2 and B3 are on PB6 and PB7.
- **Roles:** unicast and broadcast media receiver only. There is no microphone, so no call terminal.
- **Low power:** Standby is off, so PB0 keeps driving. Stop mode is still used.
- **Logs:** USART1 TX on PB12, which is the ST-LINK virtual COM port, at 115200 baud.

## Build and flash

The build needs an STM32CubeWBA checkout with its submodules (HAL, BSP, `Middlewares/ST/STM32_WPAN`)
in `CUBE_WBA`.

```bash
cmake -G Ninja -B build -DCMAKE_TOOLCHAIN_FILE=cmake/arm-gcc-cm33.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build
python ../tools/flash.py wba
```

## Use

1. On the Android phone (with LE Audio enabled), open Bluetooth settings and pair with
   `N6_Music_xxxx`.
2. Play music. The speaker switches to the music and the N6 screen goes dark after 5 s.
3. Pause. About 2.5 s later (1 s here, plus the N6's 1.5 s), the N6 wakes and beeps twice.
