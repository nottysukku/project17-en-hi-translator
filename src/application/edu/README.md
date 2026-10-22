# edu: the times-table quiz on the board

With `-DAPP_EDU_MODE=ON` the STM32N6570-DK plays the ASR_Math_Game quiz by itself, offline and without
a PC. The LCD shows a question such as "23 x 8 = ?". The child taps **SPEAK** (or presses the blue
**USER** button) and says the answer ("one eighty four") or the whole line ("twenty three times eight is
one hundred eighty four"). The board marks the answer with a large green tick or a red cross, then
asks the next question. A blurred "-ty / -teen" word ("SIXTEE") gets neither mark: the board asks "Did
you say sixty or sixteen?" and repeats the question.

Without the option, the firmware is unchanged: same transcript screen, same image size.

## Files

| File | Role |
|---|---|
| `edu_math.c/.h` | The checker: a C99 port of `ASR_Math_Game/mathspeech.py` (`parse`, `parse_many`, `may_continue`). No heap; all working memory in a caller-owned `edu_math_work_t` (about 7 KB, static). |
| `edu_math_tables.h` | Known words, misspellings and fuzzy-match targets, **generated** from `mathspeech.py` by `tools/gen_edu_tables.py`. Do not edit. |
| `edu_game.c/.h` | The quiz rules of `app.py` (quiz mode): when a turn is judged, the verdict, the score, the next question. Plain C, no RTOS. |
| `edu_display.c/.h` | The three screens (ask, listen, result), the touch screen and the USER button. Runs on the display thread. |
| `tools/gen_edu_tables.py` | Regenerates `edu_math_tables.h`; `--check` fails if it is stale. |
| `tools/stack_report.py` | Deepest call chain of the display thread, from `-fstack-usage` and the ELF's call graph. |
| `test/run_host_test.py` | The PC test: tables up to date, 252,549 vectors against Python, the quiz flow. |

Elsewhere: `display/display_thread.c` hands the recognized text to `edu_display.c` instead of the
transcript; `middleware/lcd` gained `mw_lcd_fill_rect()` and `mw_lcd_draw_line()` (the tick and cross);
the touch screen is ST's driver (`hardware/bsp/STM32N6570-DK/stm32n6570_discovery_ts.c`, `_bus.c`,
`hardware/components/gt911`, from STM32CubeN6 1.4.1), polled every 40 ms over I2C2.

## Build

From the firmware folder, with the ARM GCC of STM32CubeIDE on `PATH` (as `deploy.py` does):

```powershell
cmake --preset default -G Ninja -B build_edu_on -DCMAKE_BUILD_TYPE=Release -DSTM32_C_OPTIMIZATION=-Os -DASR_MODEL_NETWORK=cynipsnet_punct1 -DSTEDGEAI_ROOT=C:/ST/STEdgeAI/4.0/Middlewares/ST/AI -DAPP_EDU_MODE=ON
cmake --build build_edu_on --target app
```

Then sign `fsbl/build/application/fsbl.bin` and `build_edu_on/src/application/app.bin` with
`STM32_SigningTool_CLI` exactly as `cynipsnet_re/deploy/deploy.py` does.

## Flash

BOOT1 to **H**, reset, run, BOOT1 to **L**, reset. The app goes with the weights of its own model
(`cynipsnet_punct1`):

```powershell
$P = "C:\Program Files\STMicroelectronics\STM32Cube\STM32CubeProgrammer\bin"
$FW = "C:\Users\choprasu\Downloads\N6_AutoSpeech\Reverse_engineering2\Reverse_Engineering\stm32n6-speech-recognition"
& "$P\STM32_Programmer_CLI.exe" -c port=SWD freq=24000 ap=1 -el "$P\ExternalLoader\MX66UW1G45G_STM32N6570-DK.stldr" -d "$FW\build_edu_on\fsbl-trusted.bin" 0x70000000 -d "$FW\build_edu_on\app-trusted.bin" 0x70100000 -d "$FW\src\asr_engine\models\cynipsnet_punct1\network_atonbuf.xSPI2.bin" 0x71500000 -v
```

Back to the transcript: the same command with `build_cynipsnet_punct1` (or `build_edu_off`) instead of
`build_edu_on`. Each verdict is also logged on the console (`EDU 23 x 8: heard "..." -> right`).

## Keeping C and Python in step

After any change to `mathspeech.py` (a new alias, a scoring rule):

```powershell
python src\application\edu\tools\gen_edu_tables.py
python src\application\edu\test\run_host_test.py
```

The test writes its vectors into `build_edu_host/`. They cover every case of `test_mathspeech.py` (all
435 table lines, 14 phrasings, true and wrong answers, quiz and free mode, misspellings, recitals), the
recognizer's outputs in `ASR_Math_Game/eval/*.jsonl` and each of their word-by-word prefixes (the board
judges text as it arrives), and 63,000 random utterances from the checker's own vocabulary. A change to
the scoring logic itself must be made in both languages; the test shows any difference. Zig is found
from `--zig`, `$ZIG`, `PATH`, or `pip install ziglang`.

## Limits and numbers

- **Agreement:** 100 % of 252,549 vectors (template, a, b, c, verdict, ambiguity, "say it again",
  score, span, the readings shown).
- **Stack:** the display thread's deepest chain is 768 B (checker, `enumerate` recursing at most four
  deep), 980 B with an interrupt and a ThreadX context save on top. The thread has 4 KB in EDU mode
  (2 KB otherwise). Every frame is static (no variable-size arrays). Re-measure with
  `python src\application\edu\tools\stack_report.py build_edu_on` (EDU builds write the `.su` files).
  The network thread does not run the checker; it only copies text into the display mailbox, as before.
- **RAM / flash:** +11 KB RAM (8.3 KB quiz state, 2 KB more display stack), +20 KB flash.
- **Time:** about 1 µs to 30 µs per utterance on a PC, 0.9 ms for the worst fuzz case; a full 15-line
  recital about 0.3 ms.
- **Input bounds:** 1024 letters, 192 words, 240 tokens, 64 letters per word; beyond that the checker
  returns `EDU_MATH_ERR_LIMIT`. A quiz turn keeps the first 255 characters.
- **Differences from Python, outside what the board produces:** digit strings of 1000 or more are dropped
  (Python raises), and non-ASCII characters other than the multiplication sign and the right quote
  separate words.
- **Questions:** tables 2-30 times 1-15, as the PC game (defaults in `edu_game.h`); for a younger child
  configure with e.g. `-DCMAKE_C_FLAGS="-DEDU_GAME_TABLE_MAX=10 -DEDU_GAME_MULT_MAX=10"`.

## On the board

First run (2026-10-22, `build_edu_on` with `cynipsnet_punct1`): boots into the quiz, the recognizer runs
(215 ms per window), and turns are judged as on the PC ("EN TEN" for 5 x 2: right). The touch screen
must be started after the startup logo: its GT911 shares the LCD's reset line and is not ready right
after it is released (probed at once it was "not found"); `touch_start()` now runs when the logo goes and
retries, and logs the error code and which I2C2 address answers if it still fails.

Tapping SPEAK on the screen works (orientation `TS_SWAP_NONE`, as ST's example). Still to judge with
real use: the layout on the panel and how the judging delay feels (1.6 s after the last word, 3 s after
"ONE HUNDRED" or "EIGHTY").

Seen in the first turns: "one hundred and four" came out as "... HUNDRED TEN FOUR", read as 2 x 110 = 4
and crossed (the Python game does the same). A misheard "AND" is a recognizer error the checker cannot
undo safely; real recordings will show how often it happens.
