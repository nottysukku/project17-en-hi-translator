"""Build the Hindi syllable font for the STM32N6 LCD: pre-shaped Devanagari clusters, 4-bit alpha.

    python hi_font_build.py <font.ttf> <hindi_corpus.parquet> <it2 blob.json> <out.bin> [px] [weight]

Devanagari letters join, reorder and stack inside an orthographic syllable (cluster), so each
cluster is shaped with HarfBuzz and rasterised with FreeType here, on the PC. The board only splits
UTF-8 text into clusters (the same scanner as `clusters()` below, in hi_text.c), looks each one up
and blits it. Clusters missing from the table fall back to their characters one by one.

Blob layout (little-endian):
  header: "HIFT", version, px, ascent, descent, line_height, count, index_off, strings_off, bitmaps_off
  index:  count x {str_off u32, bmp_off u32, str_len u16, advance u16, width u8, height u8,
                   x_off i8, y_off i8}  sorted by the cluster's UTF-8 bytes
          (x_off: pen to bitmap left; y_off: baseline to bitmap top, negative is up)
  strings, then bitmaps: version 2 (written here) run-length codes each bitmap, see hi_font_rle.py;
  version 1 stored rows of 4-bit alpha, two pixels per byte, high nibble first, rows padded to
  whole bytes. The board reads both.
"""
import json
import os
import struct
import sys
from collections import Counter

import freetype
import numpy as np
import pandas as pd
import uharfbuzz as hb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hi_font_rle import read, write  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

CONS = set(range(0x0915, 0x093A)) | set(range(0x0958, 0x0960)) | set(range(0x0978, 0x0980))
NUKTA, HALANT = 0x093C, 0x094D
MATRA = set(range(0x093A, 0x093C)) | set(range(0x093E, 0x094D)) | {0x094E, 0x094F, 0x0955, 0x0956, 0x0957, 0x0962,
                                                                     0x0963}
MOD = {0x0900, 0x0901, 0x0902, 0x0903}
VOWEL = set(range(0x0904, 0x0915)) | {0x0960, 0x0961} | set(range(0x0972, 0x0978))
ZW = {0x200C, 0x200D}


def clusters(text):
    """Split text into Devanagari orthographic syllables; any other character stands alone."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = ord(text[i])
        j = i + 1
        if c in CONS or c in VOWEL:
            if c in CONS:
                while True:
                    if j < n and ord(text[j]) == NUKTA:
                        j += 1
                    if j < n and ord(text[j]) == HALANT:
                        k = j + 1
                        if k < n and ord(text[k]) in ZW:
                            k += 1
                        if k < n and ord(text[k]) in CONS:
                            j = k + 1
                            continue
                        j = k  # final halant (and ZWJ/ZWNJ) stays in the cluster
                    break
            elif j < n and ord(text[j]) == NUKTA:
                j += 1
            while j < n and ord(text[j]) in MATRA:
                j += 1
            if j < n and ord(text[j]) == NUKTA:
                j += 1
            while j < n and ord(text[j]) in MOD:
                j += 1
        elif c in MATRA or c in MOD or c == NUKTA or c == HALANT:
            while j < n and (ord(text[j]) in MATRA or ord(text[j]) in MOD):
                j += 1
        out.append(text[i:j])
        i = j
    return out


def main():
    font_path, corpus, it2_json, out_path = sys.argv[1:5]
    px = int(sys.argv[5]) if len(sys.argv) > 5 else 40
    weight = int(sys.argv[6]) if len(sys.argv) > 6 else 500

    counts = Counter()
    d = pd.read_parquet(corpus)
    for r in d["translation"]:
        counts.update(clusters(r["hi"]))
    total = sum(counts.values())
    keep, covered = [], 0
    for cl, n in counts.most_common():
        if covered / total >= 0.9999 and n < 20:
            break
        keep.append(cl)
        covered += n
    singles = [chr(c) for c in range(0x0900, 0x0980)] + [chr(c) for c in range(0x20, 0x7F)] + ["।", "॥"]
    keep_set = set(keep) | set(singles)
    print("corpus clusters: %d distinct; kept %d (coverage %.5f) + singles -> %d" %
          (len(counts), len(keep), covered / total, len(keep_set)))

    blob = open(font_path, "rb").read()
    face = hb.Face(blob)
    font = hb.Font(face)
    upem = face.upem
    font.scale = (upem, upem)
    font.set_variations({"wght": weight})
    ft = freetype.Face(font_path)
    axes = [a.tag for a in ft.get_variation_info().axes]
    ft.set_var_design_coords(tuple(float(weight) if t == "wght" else 100.0 for t in axes))
    ft.set_pixel_sizes(0, px)
    k = px / upem
    asc, desc = int(round(ft.size.ascender / 64)), int(round(-ft.size.descender / 64))

    entries = []
    missing = 0
    for cl in sorted(keep_set, key=lambda s: s.encode("utf-8")):
        buf = hb.Buffer()
        buf.add_str(cl)
        buf.guess_segment_properties()
        hb.shape(font, buf, {})
        pen = 0.0
        pieces = []
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
            if info.codepoint == 0 and not cl.isspace():
                missing += 1
            ft.load_glyph(info.codepoint, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_LIGHT)
            g = ft.glyph
            bm = g.bitmap
            arr = np.array(bm.buffer, dtype=np.uint8).reshape(bm.rows, bm.width) if bm.rows else np.zeros((0, 0), np.uint8)
            x = pen + pos.x_offset * k + g.bitmap_left
            y = -(pos.y_offset * k) - g.bitmap_top  # top of the glyph bitmap relative to the baseline
            pieces.append((int(round(x)), int(round(y)), arr))
            pen += pos.x_advance * k
        advance = int(round(pen))
        boxes = [(x, y, x + a.shape[1], y + a.shape[0]) for x, y, a in pieces if a.size]
        if boxes:
            x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
            x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
            canvas = np.zeros((y1 - y0, x1 - x0), np.uint8)
            for x, y, a in pieces:
                if a.size:
                    sub = canvas[y - y0:y - y0 + a.shape[0], x - x0:x - x0 + a.shape[1]]
                    np.maximum(sub, a, out=sub)
        else:
            x0, y0, canvas = 0, 0, np.zeros((0, 0), np.uint8)
        assert canvas.shape[0] < 256 and canvas.shape[1] < 256 and -128 <= x0 < 128 and -128 <= y0 < 128
        q = (canvas.astype(np.uint16) * 15 + 127) // 255
        h, w = q.shape
        packed = bytearray()
        for row in q:
            for i in range(0, w, 2):
                hi4 = int(row[i])
                lo4 = int(row[i + 1]) if i + 1 < w else 0
                packed.append((hi4 << 4) | lo4)
        entries.append((cl.encode("utf-8"), advance, w, h, x0, y0, bytes(packed)))

    header_size = 48
    index = bytearray()
    strings = bytearray()
    bitmaps = bytearray()
    for raw, adv, w, h, x0, y0, bits in entries:
        index.extend(struct.pack("<IIHHBBbb", len(strings), len(bitmaps), len(raw), adv, w, h, x0, y0))
        strings.extend(raw)
        bitmaps.extend(bits)
    index_off = header_size
    strings_off = index_off + len(index)
    bitmaps_off = strings_off + len(strings)
    line_height = int(round(px * 1.55))
    hdr = struct.pack("<4s9I", b"HIFT", 1, px, asc, desc, line_height, len(entries), index_off, strings_off, bitmaps_off)
    hdr += b"\0" * (header_size - len(hdr))
    data = hdr + index + strings + bitmaps
    data = write(*read(data), data)  # version 2: run-length coded bitmaps
    open(out_path, "wb").write(data)
    json.dump(dict(px=px, weight=weight, count=len(entries), bytes=len(data), missing_glyphs=missing),
              open(out_path + ".json", "w"))
    print("%s: %d clusters, %.2f MB, px %d, ascent %d, descent %d, line %d, missing glyphs %d" %
          (out_path, len(entries), len(data) / 1e6, px, asc, desc, line_height, missing))


if __name__ == "__main__":
    main()
