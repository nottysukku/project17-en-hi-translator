"""Pack IndicTrans2 (en -> hi) into the flat blob the STM32N6 translation engine (it2.c) reads.

    python it2_pack.py <model_dir> <hindi_vocab.json> <out.bin> [--dec-layers 0,2,4,...]

Layout (little-endian; every section 64-byte aligned):
  header (it2_header_t in it2.h)
  tensor table: n_tensors x {kind, rows, cols, data offset, scale offset}
    kind 0 = float32 vector/matrix; kind 1 = Q4: signed 4-bit, groups of 64 along a row,
    one float16 scale per group. A 32-value block is 16 bytes: byte i holds value i in its low
    nibble and value i+16 in its high nibble.
  tensors, in the fixed order of it2.c (see TENSOR ORDER below)
  source vocabulary: SentencePiece BPE pieces sorted by bytes, each {offset, length, score, dict id}
  target vocabulary: kept Hindi tokens in id order, each {offset, length}; then the strings
"""
import argparse
import json
import os
import struct

import numpy as np
from safetensors.numpy import load_file

MAGIC, VERSION, GROUP = b"IT2B", 1, 64
KIND_F32, KIND_Q4 = 0, 1


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


class Packer:
    def __init__(self):
        self.data = bytearray()
        self.table = []
        self.deq = {}

    def f32(self, name, a):
        a = np.ascontiguousarray(a, dtype=np.float32)
        align(self.data)
        off = len(self.data)
        self.data.extend(a.tobytes())
        rows, cols = (1, a.shape[0]) if a.ndim == 1 else a.shape
        self.table.append((KIND_F32, rows, cols, off, 0))
        self.deq[name] = a

    def q4(self, name, w):
        packed, scales, deq = q4(w)
        align(self.data)
        off = len(self.data)
        self.data.extend(packed)
        align(self.data)
        soff = len(self.data)
        self.data.extend(scales)
        self.table.append((KIND_Q4, w.shape[0], w.shape[1], off, soff))
        self.deq[name] = deq


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
    kept_index = {orig: i for i, orig in enumerate(keep)}

    p = Packer()
    # TENSOR ORDER (it2.c relies on it)
    p.q4("src_emb", w["model.encoder.embed_tokens.weight"])
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

    # Source vocabulary: SentencePiece BPE pieces with their scores, mapped to dict.SRC ids.
    from sentencepiece import sentencepiece_model_pb2 as pb
    spm = pb.ModelProto()
    spm.ParseFromString(open(os.path.join(args.model_dir, "model.SRC"), "rb").read())
    assert spm.trainer_spec.model_type == 2, "expected a BPE model"
    pieces = []
    for piece in spm.pieces:
        if piece.type in (1, 4):  # normal and user-defined pieces
            raw = piece.piece.encode("utf-8")
            pieces.append((raw, piece.score, src_dict.get(piece.piece, src_dict["<unk>"])))
    pieces.sort(key=lambda t: t[0])
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
    hdr = struct.pack("<4s21I", MAGIC, VERSION, 512, 8, 2048, n_enc, len(dec_layers), GROUP,
                      len(src_dict), len(keep), len(p.table),
                      src_dict["eng_Latn"], src_dict["hin_Deva"], 0, 1, 2, 3,
                      len(pieces), src_off, src_str_off, tgt_off, tgt_str_off)
    out[0:len(hdr)] = hdr
    struct.pack_into("<I", out, len(hdr), len(out))  # total size
    open(args.out, "wb").write(out)
    meta = dict(dec_layers=dec_layers, n_enc=n_enc, tgt_vocab=len(keep), src_pieces=len(pieces),
                bytes=len(out), tensors=len(p.table), kept_ids=keep)
    json.dump(meta, open(args.out + ".json", "w"))
    print("%s: %.1f MB, %d tensors, %d enc + %d dec layers, %d source pieces, %d target tokens" %
          (args.out, len(out) / 1e6, len(p.table), n_enc, len(dec_layers), len(pieces), len(keep)))



if __name__ == "__main__":
    main()
