"""Pack the MMS-TTS Hindi front end (text encoder + stochastic duration predictor) for tts_front.c.

    python tts_front_pack.py <mms_dir> <out.bin>

Layout (little-endian; sections 64-byte aligned), read in place from flash:
  header (tts_header_t in tts_front.h), then the vocabulary: 72 x {code point u32, id u32}
  tensor table at table_off: n_tensors x {kind, rows, cols, data offset, scale offset}
    kind 0 = float32; kind 1 = float16;
    kind 3 = Q8: int8 rows, groups of 64 along a row, one float16 scale per group (symmetric).
  tensors, in the fixed order of tts_front.c (TENSOR ORDER below)

Text-encoder matrices are Q8 (4-bit cost up to 20 points of Whisper CER, tts_front_eval.py);
convolutions are stored as [out, k * in], tap-major, the order tts_front.c multiplies in. The
duration predictor is float16 (368 of 369 test durations unchanged). The posterior parts of the
duration predictor and its first conv flow are not used in reverse and are left out.
"""
import json
import os
import struct
import sys

import numpy as np
from safetensors.numpy import load_file

MAGIC, VERSION, GROUP = b"TTSF", 1, 64
KIND_F32, KIND_F16, KIND_Q8 = 0, 1, 3
HEADER_SIZE = 128


def align(buf, n=64):
    buf.extend(b"\0" * ((-len(buf)) % n))


def q8(w):
    """Symmetric int8, one float16 scale per group of 64 along each row."""
    rows, cols = w.shape
    assert cols % GROUP == 0, w.shape
    g = w.reshape(rows, cols // GROUP, GROUP).astype(np.float32)
    s = (np.abs(g).max(axis=2, keepdims=True) / 127.0).astype(np.float16).astype(np.float32)
    s[s == 0] = 1.0
    q = np.clip(np.round(g / s), -127, 127).astype(np.int8)
    return q.reshape(rows, cols).tobytes(), s.reshape(rows, cols // GROUP).astype(np.float16).tobytes()


class Packer:
    def __init__(self):
        self.data = bytearray()
        self.table = []

    def add(self, a, kind):
        a = np.asarray(a, np.float32)
        rows, cols = (1, a.shape[0]) if a.ndim == 1 else a.shape
        align(self.data)
        off, soff = len(self.data), 0
        if kind == KIND_F32:
            self.data.extend(a.astype("<f4").tobytes())
        elif kind == KIND_F16:
            self.data.extend(a.astype("<f2").tobytes())
        else:
            q, s = q8(a.reshape(rows, cols))
            self.data.extend(q)
            align(self.data)
            soff = len(self.data)
            self.data.extend(s)
        self.table.append((kind, rows, cols, off, soff))


def conv_matrix(w):
    """Conv1d weight [out, in, k] -> [out, k * in], tap-major."""
    return w.transpose(0, 2, 1).reshape(w.shape[0], -1)


def main():
    mms, out_path = sys.argv[1], sys.argv[2]
    cfg = json.load(open(os.path.join(mms, "config.json")))
    vocab = json.load(open(os.path.join(mms, "vocab.json"), encoding="utf-8"))
    w = load_file(os.path.join(mms, "model.safetensors"))
    assert cfg["hidden_size"] == 192 and cfg["num_hidden_layers"] == 6 and cfg["num_attention_heads"] == 2
    assert cfg["ffn_dim"] == 768 and cfg["ffn_kernel_size"] == 3 and cfg["window_size"] == 4
    assert cfg["duration_predictor_flow_bins"] == 10 and cfg["depth_separable_num_layers"] == 3
    assert cfg["hidden_act"] == "relu" and cfg["use_stochastic_duration_prediction"]

    p = Packer()
    te = "text_encoder."
    # TENSOR ORDER (tts_front.c relies on it)
    p.add(w[te + "embed_tokens.weight"], KIND_F32)
    for i in range(6):
        q = te + "encoder.layers.%d." % i
        for n in ("q_proj", "k_proj", "v_proj", "out_proj"):
            p.add(w[q + "attention.%s.weight" % n], KIND_Q8)
            p.add(w[q + "attention.%s.bias" % n], KIND_F32)
        p.add(w[q + "attention.emb_rel_k"][0], KIND_F32)
        p.add(w[q + "attention.emb_rel_v"][0], KIND_F32)
        p.add(w[q + "layer_norm.weight"], KIND_F32)
        p.add(w[q + "layer_norm.bias"], KIND_F32)
        p.add(conv_matrix(w[q + "feed_forward.conv_1.weight"]), KIND_Q8)
        p.add(w[q + "feed_forward.conv_1.bias"], KIND_F32)
        p.add(conv_matrix(w[q + "feed_forward.conv_2.weight"]), KIND_Q8)
        p.add(w[q + "feed_forward.conv_2.bias"], KIND_F32)
        p.add(w[q + "final_layer_norm.weight"], KIND_F32)
        p.add(w[q + "final_layer_norm.bias"], KIND_F32)
    p.add(conv_matrix(w[te + "project.weight"]), KIND_Q8)
    p.add(w[te + "project.bias"], KIND_F32)

    dp = "duration_predictor."

    def dds(prefix):
        for i in range(3):
            p.add(w[prefix + "convs_dilated.%d.weight" % i][:, 0, :], KIND_F16)  # [192, 3]
            p.add(w[prefix + "convs_dilated.%d.bias" % i], KIND_F16)
            p.add(w[prefix + "norms_1.%d.weight" % i], KIND_F16)
            p.add(w[prefix + "norms_1.%d.bias" % i], KIND_F16)
            p.add(w[prefix + "convs_pointwise.%d.weight" % i][:, :, 0], KIND_F16)
            p.add(w[prefix + "convs_pointwise.%d.bias" % i], KIND_F16)
            p.add(w[prefix + "norms_2.%d.weight" % i], KIND_F16)
            p.add(w[prefix + "norms_2.%d.bias" % i], KIND_F16)

    p.add(w[dp + "conv_pre.weight"][:, :, 0], KIND_F16)
    p.add(w[dp + "conv_pre.bias"], KIND_F16)
    dds(dp + "conv_dds.")
    p.add(w[dp + "conv_proj.weight"][:, :, 0], KIND_F16)
    p.add(w[dp + "conv_proj.bias"], KIND_F16)
    for f in (4, 3, 2):  # the conv flows in the order the reverse pass uses them
        q = dp + "flows.%d." % f
        p.add(w[q + "conv_pre.weight"][:, 0, 0], KIND_F16)
        p.add(w[q + "conv_pre.bias"], KIND_F16)
        dds(q + "conv_dds.")
        p.add(w[q + "conv_proj.weight"][:, :, 0], KIND_F16)  # [29, 192]
        p.add(w[q + "conv_proj.bias"], KIND_F16)
    p.add(w[dp + "flows.0.translate"][:, 0], KIND_F32)
    p.add(w[dp + "flows.0.log_scale"][:, 0], KIND_F32)

    vocab_bytes = b"".join(struct.pack("<II", ord(ch), i) for ch, i in sorted(vocab.items(), key=lambda kv: kv[1]))
    out = bytearray(b"\0" * HEADER_SIZE)
    vocab_off = len(out)
    out.extend(vocab_bytes)
    align(out)
    table_off = len(out)
    out.extend(b"\0" * (20 * len(p.table)))
    align(out)
    tensors_off = len(out)
    out.extend(p.data)
    align(out)
    table = b"".join(struct.pack("<5I", k, r, c, tensors_off + o, tensors_off + s if k == KIND_Q8 else 0)
                     for k, r, c, o, s in p.table)
    out[table_off:table_off + len(table)] = table
    hdr = struct.pack("<4s11I3f", MAGIC, VERSION, 192, 2, 768, 6, 4, len(vocab), vocab_off, len(p.table), table_off,
                      len(out), cfg["noise_scale"], cfg["noise_scale_duration"], 1.0 / cfg["speaking_rate"])
    out[0:len(hdr)] = hdr
    open(out_path, "wb").write(out)
    print("%s: %.2f MB, %d tensors, %d symbols" % (out_path, len(out) / 1e6, len(p.table), len(vocab)))


if __name__ == "__main__":
    main()
