"""Render text from a hi_font blob the way hi_text.c will (cluster lookup + blit) into a PNG."""
import struct
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from hi_font_build import clusters  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
blob = open(sys.argv[1], "rb").read()
magic, ver, px, asc, desc, line_h, count, idx_off, str_off, bmp_off = struct.unpack_from("<4s9I", blob, 0)
table = {}
for i in range(count):
    so, bo, sl, adv, w, h, x0, y0 = struct.unpack_from("<IIHHBBbb", blob, idx_off + 16 * i)
    table[blob[str_off + so:str_off + so + sl].decode("utf-8")] = (bo, adv, w, h, x0, y0)
missing = [k for k, v in table.items() if v[2] == 0 and not k.isspace()]
print("empty glyphs:", len(missing), "".join(sorted(missing))[:200])

LINES = ["नमस्ते, आप कैसे हैं?", "मुझे समझ नहीं आ रहा है कि आप क्या कह रहे हैं।",
         "क्या आप रेलवे स्टेशन खोजने में मेरी मदद कर सकते हैं?", "कृपया बत्तियाँ बंद कर दें। धन्यवाद! श्री, क्षमा, ज्ञान, द्वार",
         "आपकी मदद के लिए बहुत-बहुत धन्यवाद। मैं कल बाज़ार जा रहा हूँ।"]
W, H = 800, line_h * len(LINES) + 20
img = np.full((H, W), 255, np.float32)
fallbacks = 0
for li, text in enumerate(LINES):
    x, base = 10, 10 + asc + li * line_h
    for cl in clusters(text):
        parts = [cl] if cl in table else list(cl)
        fallbacks += cl not in table
        for p in parts:
            if p not in table:
                continue
            bo, adv, w, h, x0, y0 = table[p]
            if w:
                row_bytes = (w + 1) // 2
                for r in range(h):
                    for c in range(w):
                        b = blob[bmp_off + bo + r * row_bytes + c // 2]
                        a = ((b >> 4) if c % 2 == 0 else (b & 15)) / 15.0
                        yy, xx = base + y0 + r, x + x0 + c
                        if 0 <= yy < H and 0 <= xx < W:
                            img[yy, xx] = min(img[yy, xx], 255 * (1 - a))
            x += adv
Image.fromarray(img.astype(np.uint8)).save(sys.argv[2])
print("clusters not in the table (drawn letter by letter):", fallbacks)
