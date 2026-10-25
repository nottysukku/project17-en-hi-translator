#!/bin/sh
# st_compile_asr.sh <export_dir> [name]: compile the fine-tuned recognizer for the DK and put it into
# the firmware. Git Bash on Windows, from the repository folder.
#
#   export_dir  the folder the Colab notebook wrote (Google Drive: project17_asr/export/<name>):
#               <name>.onnx, network_shape.h, provenance.json
#   name        default cynipsnet_inen1
#
# The ST Edge AI command is the one cynipsnet_punct1 was compiled with (its report, line 3): int8
# input and output, channel-last, the same neural_art.json and memory-pool file. So the weights
# land at 0x71500000 again and the activations in the same RAM, and nothing else moves.
# Afterwards: firmware-n6/src/asr_engine/models/<name>/ holds the generated code and weights, and
# models.json points the "asr" blob at them.
# ST_ROUNDTRIP: folder with neural_art.json and stm32n6.mpool (default below, the owner's PC).
set -e
E=$(realpath "$1")
NAME=${2:-cynipsnet_inen1}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
RT=${ST_ROUNDTRIP:-/c/Users/choprasu/Downloads/N6_AutoSpeech/Reverse_engineering2/Reverse_Engineering/cynipsnet_re/roundtrip}
W="$REPO/model-tools/out/st/$NAME"
DST="$REPO/firmware-n6/src/asr_engine/models/$NAME"
[ -f "$E/$NAME.onnx" ] || { echo "no $E/$NAME.onnx"; exit 1; }
[ -f "$RT/neural_art.json" ] || { echo "no $RT/neural_art.json (set ST_ROUNDTRIP)"; exit 1; }
rm -rf "$W"; mkdir -p "$W"; cp "$RT/neural_art.json" "$RT/stm32n6.mpool" "$W/"
cd "$W"
"/c/ST/STEdgeAI/4.0/Utilities/windows/stedgeai.exe" generate -m "$(cygpath -w "$E/$NAME.onnx")" --target stm32n6 \
  --st-neural-art "default@$(cygpath -w "$W/neural_art.json")" --input-data-type int8 --output-data-type int8 \
  --inputs-ch-position chlast --outputs-ch-position chlast -c lossless --c-api st-ai -o st_out -w st_ws \
  > gen.log 2>&1 || { tail -30 gen.log; exit 1; }
echo "== report (expect: all epochs pure hardware, octoFlash from 0x71500000, npuRAM5 from 0x342E0000)"
grep -E "Total number of epochs|pure software|hybrid|pure hardware|octoFlash|hyperRAM \[|npuRAM5" \
  st_out/network_generate_report.txt
echo "== input and output quantization (input must be 0.0628561676, output as in network_shape.h)"
grep -A1 -E "define STAI_NETWORK_(IN|OUT)_1_SCALES" st_out/stai_network.h | grep -v "^--"
grep -E "OUTPUT_SCALE" "$E/network_shape.h"
mkdir -p "$DST"
for f in network.c network.h stai_network.c stai_network.h network_atonbuf.xSPI2.bin network_generate_report.txt; do
  if [ -f "st_out/$f" ]; then cp "st_out/$f" "$DST/"; else echo "WARNING: st_out/$f was not generated"; fi
done
cp "$E/network_shape.h" "$E/provenance.json" "$DST/"
cp "$REPO/firmware-n6/src/asr_engine/models/cynipsnet_punct1/CMakeLists.txt" "$DST/"
python - "$REPO" "$NAME" <<'EOF'
import json, sys
repo, name = sys.argv[1], sys.argv[2]
path = repo + "/firmware-n6/src/application/translator/models.json"
with open(path) as f:
    text = f.read()
old = "firmware-n6/src/asr_engine/models/cynipsnet_punct1/network_atonbuf.xSPI2.bin"
new = "firmware-n6/src/asr_engine/models/%s/network_atonbuf.xSPI2.bin" % name
text = text.replace(old, new).replace("cynipsnet_punct1; address fixed", "%s; address fixed" % name)
text = text.replace('"make": "committed (cynipsnet_punct1)"',
                    '"make": "model-tools/asr_finetune (Colab) + st_compile_asr.sh"')
with open(path, "w") as f:
    f.write(text)
json.loads(text)
print("models.json: the asr blob is now", new)
EOF
ls -l "$DST"
