#!/bin/sh
# st_compile.sh <model.onnx> <work_dir> [mpool]: ST Edge AI for the STM32N6 NPU, then HW/SW epoch counts.
set -e
M=$(cygpath -w "$(realpath "$1")"); W=$2; RT=/c/Users/choprasu/Downloads/N6_AutoSpeech/Reverse_engineering2/Reverse_Engineering/cynipsnet_re/roundtrip
POOL=${3:-$RT/stm32n6.mpool}
rm -rf "$W"; mkdir -p "$W"; cp "$RT/neural_art.json" "$W/"; cp "$POOL" "$W/stm32n6.mpool"
cd "$W"
"/c/ST/STEdgeAI/4.0/Utilities/windows/stedgeai.exe" generate -m "$M" --target stm32n6 \
  --st-neural-art "default@$(cygpath -w "$W/neural_art.json")" --input-data-type float32 --output-data-type float32 \
  -c lossless --c-api st-ai -o st_out -w st_ws > gen.log 2>&1 || { tail -30 gen.log; exit 1; }
grep -E "macc=|pure software|hybrid|pure hardware|octoFlash|hyperRAM \[" st_out/network_generate_report.txt
grep -oE "\| *(HW|SW) *\| *[A-Za-z()]*" gen.log | sed 's/  */ /g' | sort | uniq -c | sort -rn | head -12
