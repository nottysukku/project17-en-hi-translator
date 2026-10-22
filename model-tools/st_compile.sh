#!/bin/sh
# st_compile.sh <model.onnx> <work_dir> [mpool] [name]: ST Edge AI 4.0 for the STM32N6 NPU, then
# the HW/SW epoch counts. Git Bash on Windows.
#   mpool  memory-pool file (default: the one the recognizer was compiled with); for the voice
#          networks make it with mpool_edit.py so their flash and PSRAM match models.json
#   name   network name (stedgeai --name): "vocoder" and "flow" for the voice; files and C
#          functions are then vocoder_*.bin, stai_vocoder_*() and so on
# ST_ROUNDTRIP: folder with neural_art.json and stm32n6.mpool (default below, the owner's PC).
set -e
M=$(cygpath -w "$(realpath "$1")"); W=$(realpath -m "$2")
RT=${ST_ROUNDTRIP:-/c/Users/choprasu/Downloads/N6_AutoSpeech/Reverse_engineering2/Reverse_Engineering/cynipsnet_re/roundtrip}
POOL=${3:-$RT/stm32n6.mpool}
NAME=${4:-network}
rm -rf "$W"; mkdir -p "$W"; cp "$RT/neural_art.json" "$W/"; cp "$POOL" "$W/stm32n6.mpool"
cd "$W"
"/c/ST/STEdgeAI/4.0/Utilities/windows/stedgeai.exe" generate -m "$M" --target stm32n6 --name "$NAME" \
  --st-neural-art "default@$(cygpath -w "$W/neural_art.json")" --input-data-type float32 --output-data-type float32 \
  -c lossless --c-api st-ai -o st_out -w st_ws > gen.log 2>&1 || { tail -30 gen.log; exit 1; }
grep -E "macc=|pure software|hybrid|pure hardware|octoFlash|hyperRAM \[" st_out/*_generate_report.txt
grep -oE "\| *(HW|SW) *\| *[A-Za-z()]*" gen.log | sed 's/  */ /g' | sort | uniq -c | sort -rn | head -12
ls -l st_out/*.bin
