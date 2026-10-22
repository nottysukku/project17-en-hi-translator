"""Pack IndicTrans2 (en -> hi) into the flat blob the STM32N6 translation engine (it2.c) reads.

    python it2_pack.py <model_dir> <hindi_vocab.json> <out.bin> [--dec-layers 0,2,4,...] [--src-keep N]

Layout (little-endian; every section 64-byte aligned):
  header (it2_header_t in it2.h)
  tensor table: n_tensors x {kind, rows, cols, data offset, scale offset}
    kind 0 = float32 vector/matrix; kind 1 = Q4: signed 4-bit, groups of 64 along a row,
    one float16 scale per group. A 32-value block is 16 bytes: byte i holds value i in its low
    nibble and value i+16 in its high nibble.
    kind 2 = Q4 with 8-bit scales (--scale8): the same 4-bit data; the scale area holds one
    float32 base for the tensor, then one byte per group: scale = base * 2^(code / 16). The
    code is rounded up and the weights are quantized against the rounded scale.
  tensors, in the fixed order of it2.c (see TENSOR ORDER below)
  source vocabulary: SentencePiece BPE pieces sorted by bytes, each {offset, length, score, id}
    id = the piece's row in the source embedding; or, if --src-keep dropped that row,
    SRC_SPLIT | (byte length of the left part): the engine then emits the two pieces the BPE
    merged into this one, recursively (SentencePiece's rule for a restricted vocabulary).
  target vocabulary: kept Hindi tokens in id order, each {offset, length}; then the strings

Version 2 adds SRC_SPLIT entries; a blob without --src-keep is otherwise the same as version 1.
Version 3 (--scale8) has kind-2 tensors.
"""
import argparse
import json
import os
import struct

import numpy as np
from safetensors.numpy import load_file

MAGIC, VERSION, GROUP = b"IT2B", 2, 64
SRC_SPLIT = 0x80000000
KIND_F32, KIND_Q4, KIND_Q4S8 = 0, 1, 2
SCALE_STEPS = 16  # kind 2: scale codes per octave


def pow2_steps():
    """2^(i/16) for i < 256 as float32, built the way it2.c builds its table (exact powers of two
    times 16 float32 constants), so both sides use the same scales."""
    frac = np.array([2.0 ** (k / SCALE_STEPS) for k in range(SCALE_STEPS)], np.float32)
    return np.array([np.ldexp(frac[i % SCALE_STEPS], i // SCALE_STEPS) for i in range(256)], np.float32)


POW2 = pow2_steps()


def align(buf, n=64):
    buf.extend(b"\0" * ((-len(buf)) % n))


def q4(w):
    """Symmetric 4-bit, per 64-group scales (float16). Returns (packed bytes, scales bytes, dequant)."""
    rows, cols = w.shape
    assert cols % GROUP == 0
    g = w.reshape(rows, cols // GROUP, GROUP).astype(np.float32)
    scale = (np.abs(g).max(axis=2, keepdims=True) / 7.0).astype(np.float16).astype(np.float32)
    scale[scale == 0] = 1e-8
    q = np.clip(np.round(g / scale), -7, 7).astype(np.int8)
    deq = (q.astype(np.float32) * scale).reshape(rows, cols)
    blocks = q.reshape(rows, cols // 32, 2, 16)            # [.., lo(0..15) / hi(16..31), 16]
    lo, hi = blocks[:, :, 0, :], blocks[:, :, 1, :]
    packed = ((lo.astype(np.uint8) & 0x0F) | ((hi.astype(np.uint8) & 0x0F) << 4)).reshape(rows, cols // 2)
    return packed.tobytes(), scale.reshape(rows, cols // GROUP).astype(np.float16).tobytes(), deq


def q4_s8(w):
    """4-bit like q4(), with 8-bit log scales. Returns (packed bytes, scale area bytes, dequant)."""
    rows, cols = w.shape
    g = w.reshape(rows, cols // GROUP, GROUP).astype(np.float32)
    ideal = np.abs(g).max(axis=2) / 7.0
    base = np.float32(max(float(ideal.max()), 1e-30) * 2.0 ** (-255.0 / SCALE_STEPS))
    with np.errstate(divide="ignore"):
        code = np.ceil(SCALE_STEPS * np.log2(np.maximum(ideal, 1e-38) / base) - 1e-6)
    code = np.clip(code, 0, 255).astype(np.uint8)
    scale = (base * POW2[code])[:, :, None]
    q = np.clip(np.round(g / scale), -7, 7).astype(np.int8)
    deq = (q.astype(np.float32) * scale).reshape(rows, cols)
    blocks = q.reshape(rows, cols // 32, 2, 16)
    lo, hi = blocks[:, :, 0, :], blocks[:, :, 1, :]
    packed = ((lo.astype(np.uint8) & 0x0F) | ((hi.astype(np.uint8) & 0x0F) << 4)).reshape(rows, cols // 2)
    return packed.tobytes(), struct.pack("<f", float(base)) + code.tobytes(), deq


class Packer:
    def __init__(self, scale8=False):
        self.data = bytearray()
        self.table = []
        self.deq = {}
        self.scale8 = scale8

    def f32(self, name, a):
        a = np.ascontiguousarray(a, dtype=np.float32)
        align(self.data)
        off = len(self.data)
        self.data.extend(a.tobytes())
        rows, cols = (1, a.shape[0]) if a.ndim == 1 else a.shape
        self.table.append((KIND_F32, rows, cols, off, 0))
        self.deq[name] = a

    def q4(self, name, w):
        packed, scales, deq = (q4_s8 if self.scale8 else q4)(w)
        align(self.data)
        off = len(self.data)
        self.data.extend(packed)
        align(self.data)
        soff = len(self.data)
        self.data.extend(scales)
        self.table.append((KIND_Q4S8 if self.scale8 else KIND_Q4, w.shape[0], w.shape[1], off, soff))
        self.deq[name] = deq


def last_merge(piece, score):
    """Byte length of the left part of the last BPE merge that builds `piece` from its characters,
    or None if the BPE never builds it. Within a word the merges inside a piece's span happen in
    the same order as for the piece alone, so this split is the same wherever the piece occurs."""
    syms, last = list(piece), None
    while len(syms) > 1:
        best, best_score = -1, None
        for i in range(len(syms) - 1):
            sc = score.get(syms[i] + syms[i + 1])
            if sc is not None and (best_score is None or sc > best_score):
                best, best_score = i, sc
        if best < 0:
            break
        last = syms[best]
        syms[best:best + 2] = [syms[best] + syms[best + 1]]
    return len(last.encode("utf-8")) if (syms == [piece] and last is not None) else None


def source_vocab(model_dir, src_dict, keep_n):
    """Source embedding rows to keep and the tokeniser table.

    keep_n = 0 keeps every row. Otherwise the special tokens, the two language tags, every
    single-character piece and the most frequent pieces up to keep_n rows are kept. dict.SRC is
    fairseq's dictionary, sorted by frequency in IndicTrans2's training data, so its id order is
    the frequency ranking. Rows the tokeniser can never produce (dict entries that are not
    SentencePiece pieces, such as stray characters) are dropped first.
    Returns (rows: original ids in row order, pieces: sorted [(bytes, score, id)], stats).
    """
    from sentencepiece import sentencepiece_model_pb2 as pb
    spm = pb.ModelProto()
    spm.ParseFromString(open(os.path.join(model_dir, "model.SRC"), "rb").read())
    assert spm.trainer_spec.model_type == 2, "expected a BPE model"
    table = [(pc.piece, pc.score) for pc in spm.pieces if pc.type in (1, 4)]  # normal and user-defined
    unk = src_dict["<unk>"]
    if keep_n:
        must = {0, 1, 2, 3, src_dict["eng_Latn"], src_dict["hin_Deva"]}
        must |= {src_dict[t] for t, _ in table if len(t) == 1 and t in src_dict}
        ranked = sorted({src_dict[t] for t, _ in table if t in src_dict} - must)
        rows = sorted(must | set(ranked[:max(0, keep_n - len(must))]))
    else:
        rows = sorted(src_dict.values())
    row_of = {orig: r for r, orig in enumerate(rows)}
    score = dict(table)
    pieces, split, unreachable = [], 0, 0
    for t, sc in table:
        if t not in src_dict:
            pid = row_of[unk]
        elif src_dict[t] in row_of:
            pid = row_of[src_dict[t]]
        else:
            left = last_merge(t, score)
            if left is None:
                unreachable += 1
                continue  # never built by the BPE: leaving it out changes nothing
            pid = SRC_SPLIT | left
            split += 1
        pieces.append((t.encode("utf-8"), sc, pid))
    pieces.sort(key=lambda e: e[0])
    return rows, pieces, dict(split=split, unreachable=unreachable)


def linear(p, w, prefix, name):
    p.q4(prefix + name + ".weight", w[prefix + name + ".weight"])
    p.f32(prefix + name + ".bias", w[prefix + name + ".bias"])


def ln(p, w, prefix):
    p.f32(prefix + ".weight", w[prefix + ".weight"])
    p.f32(prefix + ".bias", w[prefix + ".bias"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("vocab_json")
    ap.add_argument("out")
    ap.add_argument("--dec-layers", default="0,2,4,6,8,10,12,14,16,17")
    ap.add_argument("--src-keep", type=int, default=0,
                    help="keep only the N most frequent source embedding rows (0: all)")
    ap.add_argument("--scale8", action="store_true", help="8-bit log scales instead of float16 (kind 2)")
    args = ap.parse_args()
    w = load_file(os.path.join(args.model_dir, "model.safetensors"))
    dec_layers = [int(x) for x in args.dec_layers.split(",")]
    n_enc = len({k.split(".")[3] for k in w if k.startswith("model.encoder.layers.")})
    keep = json.load(open(args.vocab_json))
    src_dict = json.load(open(os.path.join(args.model_dir, "dict.SRC.json"), encoding="utf-8"))
    tgt_dict = json.load(open(os.path.join(args.model_dir, "dict.TGT.json"), encoding="utf-8"))
    tgt_inv = {v: k for k, v in tgt_dict.items()}
    for must in (0, 1, 2, 3):
        assert must in keep
    src_rows, pieces, src_stats = source_vocab(args.model_dir, src_dict, args.src_keep)
    src_row_of = {orig: r for r, orig in enumerate(src_rows)}

    p = Packer(args.scale8)
    # TENSOR ORDER (it2.c relies on it)
    p.q4("src_emb", w["model.encoder.embed_tokens.weight"][src_rows])
    ln(p, w, "model.encoder.layernorm_embedding")
    ln(p, w, "model.encoder.layer_norm")
    for i in range(n_enc):
        q = "model.encoder.layers.%d." % i
        ln(p, w, q + "self_attn_layer_norm")
        for n in ("q_proj", "k_proj", "v_proj", "out_proj"):
            linear(p, w, q + "self_attn.", n)
        ln(p, w, q + "final_layer_norm")
        linear(p, w, q, "fc1")
        linear(p, w, q, "fc2")
    p.q4("tgt_emb", w["model.decoder.embed_tokens.weight"][keep])
    ln(p, w, "model.decoder.layernorm_embedding")
    ln(p, w, "model.decoder.layer_norm")
    for i in dec_layers:
        q = "model.decoder.layers.%d." % i
        ln(p, w, q + "self_attn_layer_norm")
        for n in ("q_proj", "k_proj", "v_proj", "out_proj"):
            linear(p, w, q + "self_attn.", n)
        ln(p, w, q + "encoder_attn_layer_norm")
        for n in ("q_proj", "k_proj", "v_proj", "out_proj"):
            linear(p, w, q + "encoder_attn.", n)
        ln(p, w, q + "final_layer_norm")
        linear(p, w, q, "fc1")
        linear(p, w, q, "fc2")

    # Source vocabulary: SentencePiece BPE pieces with their scores and embedding rows.
    src_strings = bytearray()
    src_entries = bytearray()
    for raw, score, sid in pieces:
        src_entries.extend(struct.pack("<IIfI", len(src_strings), len(raw), score, sid))
        src_strings.extend(raw)
    tgt_strings = bytearray()
    tgt_entries = bytearray()
    for orig in keep:
        raw = tgt_inv[orig].encode("utf-8")
        tgt_entries.extend(struct.pack("<II", len(tgt_strings), len(raw)))
        tgt_strings.extend(raw)

    header_size = 256
    table_size = 20 * len(p.table)
    out = bytearray(b"\0" * header_size)
    out.extend(struct.pack("<%dI" % (5 * len(p.table)), *[0] * (5 * len(p.table))))
    align(out)
    tensors_off = len(out)
    out.extend(p.data)
    align(out)
    src_off = len(out)
    out.extend(src_entries)
    align(out)
    src_str_off = len(out)
    out.extend(src_strings)
    align(out)
    tgt_off = len(out)
    out.extend(tgt_entries)
    align(out)
    tgt_str_off = len(out)
    out.extend(tgt_strings)
    align(out)

    table = bytearray()
    for kind, rows, cols, off, soff in p.table:
        table.extend(struct.pack("<5I", kind, rows, cols, tensors_off + off, tensors_off + soff if kind else 0))
    out[header_size:header_size + table_size] = table
    hdr = struct.pack("<4s21I", MAGIC, 3 if args.scale8 else VERSION, 512, 8, 2048, n_enc, len(dec_layers), GROUP,
                      len(src_rows), len(keep), len(p.table),
                      src_row_of[src_dict["eng_Latn"]], src_row_of[src_dict["hin_Deva"]], 0, 1, 2, 3,
                      len(pieces), src_off, src_str_off, tgt_off, tgt_str_off)
    out[0:len(hdr)] = hdr
    struct.pack_into("<I", out, len(hdr), len(out))  # total size
    open(args.out, "wb").write(out)
    meta = dict(dec_layers=dec_layers, n_enc=n_enc, tgt_vocab=len(keep), src_pieces=len(pieces),
                src_vocab=len(src_rows), bytes=len(out), tensors=len(p.table), kept_ids=keep, scale8=args.scale8,
                src_rows=src_rows)
    json.dump(meta, open(args.out + ".json", "w"))
    print("%s: %.1f MB, %d tensors, %d enc + %d dec layers, %d source pieces (%d split, %d left out), "
          "%d source rows, %d target tokens" %
          (args.out, len(out) / 1e6, len(p.table), n_enc, len(dec_layers), len(pieces), src_stats["split"],
           src_stats["unreachable"], len(src_rows), len(keep)))



if __name__ == "__main__":
    main()
