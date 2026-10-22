"""Run-length coding of the Hindi syllable font (HIFT version 2), and a v1 -> v2 converter.

    python hi_font_rle.py <in.bin> <out.bin>      convert (v1 or v2 in), check, report sizes

Version 1 stores each syllable bitmap as rows of 4-bit alpha, two pixels per byte. Version 2 stores
the same pixels, row after row without padding, as runs of one value: each byte is
value << 4 | (run length - 1), runs of 1 to 16 pixels, running on across rows. Antialiased text is
mostly empty or solid, so this halves the font; the board decodes one row at a time
(xl_hindi_text.c). Everything else in the blob (header, index, strings) is unchanged apart from
the version number and the bitmap offsets.
"""
import struct
import sys

HEADER = "<4s9I"  # magic, version, px, ascent, descent, line_height, count, index_off, strings_off, bitmaps_off
ENTRY = "<IIHHBBbb"  # str_off, bmp_off, str_len, advance, width, height, x_off, y_off


def encode(pixels):
    """Alpha values 0..15 (row-major, no padding) -> run bytes."""
    out = bytearray()
    i, n = 0, len(pixels)
    while i < n:
        j = i
        while j < n and pixels[j] == pixels[i] and j - i < 16:
            j += 1
        out.append((pixels[i] << 4) | (j - i - 1))
        i = j
    return bytes(out)


def decode(data, count):
    """Run bytes -> count alpha values, and the number of bytes used."""
    out, k = [], 0
    while len(out) < count:
        out.extend([data[k] >> 4] * ((data[k] & 15) + 1))
        k += 1
    assert len(out) == count, "a run crosses the end of the bitmap"
    return out, k


def unpack(packed, w, h):
    """Version-1 rows (two pixels per byte, high nibble first, rows padded) -> alpha values."""
    rb = (w + 1) // 2
    px = []
    for r in range(h):
        row = packed[r * rb:(r + 1) * rb]
        for c in range(w):
            px.append(row[c // 2] >> 4 if c % 2 == 0 else row[c // 2] & 15)
    return px


def read(blob):
    """-> (header fields, [(entry fields, alpha values)])."""
    hdr = list(struct.unpack_from(HEADER, blob, 0))
    magic, version, _, _, _, _, count, index_off, _, bitmaps_off = hdr
    assert magic == b"HIFT" and version in (1, 2)
    glyphs = []
    for i in range(count):
        e = struct.unpack_from(ENTRY, blob, index_off + 16 * i)
        w, h, start = e[4], e[5], bitmaps_off + e[1]
        if version == 1:
            px = unpack(blob[start:start + ((w + 1) // 2) * h], w, h)
        else:
            px, _ = decode(blob[start:start + 2 * w * h + 1], w * h)
        glyphs.append((e, px))
    return hdr, glyphs


def write(hdr, glyphs, blob):
    """Version-2 blob with the header, index and strings of `blob` and the glyphs run-length coded."""
    index_off, strings_off, bitmaps_off = hdr[7], hdr[8], hdr[9]
    index = bytearray()
    bitmaps = bytearray()
    for e, px in glyphs:
        index.extend(struct.pack(ENTRY, e[0], len(bitmaps), *e[2:]))
        bitmaps.extend(encode(px))
    head = bytearray(blob[:index_off])
    struct.pack_into("<I", head, 4, 2)
    return bytes(head + index + blob[strings_off:bitmaps_off] + bitmaps)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    blob = open(src, "rb").read()
    hdr, glyphs = read(blob)
    out = write(hdr, glyphs, blob)
    _, again = read(out)
    assert [p for _, p in again] == [p for _, p in glyphs], "round trip changed a pixel"
    open(dst, "wb").write(out)
    print("%s: version %d, %d syllables, %.2f MB -> %s: version 2, %.2f MB (pixels identical)" %
          (src, hdr[1], len(glyphs), len(blob) / 1e6, dst, len(out) / 1e6))


if __name__ == "__main__":
    main()
