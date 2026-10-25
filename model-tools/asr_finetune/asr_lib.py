"""Fine-tune the DK's speech recognizer (CynipsNet, CTC) for Indian-accented English.

Everything the Colab notebook runs lives here; the notebook writes this file to /content and
imports it, and model-tools/asr_finetune/smoke_test.py runs the same code on a PC.

The board decides the rules, so nothing here may drift from the firmware:
- features: firmware-n6/src/asr_engine/mel/mel.c and mel_tables.c (pre-emphasis 0.97, 20 ms
  symmetric Hann frames every 10 ms, 512-point FFT, power, 64 Slaney mel bands 0-8 kHz,
  ln(x + 2^-24)); checked against the C code by model-tools/asr_feature_check.py;
- normalization: norm.c (per band mean and deviation, int8 input with scale 0.0628561676);
- windows: 600 frames in, 300 steps out, the decoder commits steps 210-234 of each window
  (asr_model_params.h, net_ctc.c);
- classes: "_ ABCDEFGHIJKLMNOPQRSTUVWXYZ',.?" (blank, space, letters, apostrophe, marks).

The network itself is not rebuilt from a guess: it is read from the QDQ ONNX file that was
compiled for the board (cynipsnet_punct1.onnx), trained in PyTorch, and written back into the
same graph, so the new model has exactly the old layers, size and memory placement.
"""
import io
import json
import math
import os
import random
import re
import time
import unicodedata
from dataclasses import dataclass, field
from fnmatch import fnmatch

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# ----------------------------------------------------------------------------------------------
# Board constants (firmware-n6/src/asr_engine/asr_model_params.h, network_shape.h)
# ----------------------------------------------------------------------------------------------
TOKENS = "_ ABCDEFGHIJKLMNOPQRSTUVWXYZ',.?"
BLANK, SPACE, PUNCT_FIRST = 0, 1, 29
SR, FRAME, HOP, NFFT, BANDS = 16000, 320, 160, 512, 64
PREEMPHASIS = 0.97
LOG_GUARD = 5.9604644775390625e-08  # 2^-24
INPUT_SCALE, INPUT_ZP = 0.0628561676, 0
WIN_FRAMES, WIN_STEPS = 600, 300
LEFT_FRAMES, STRIDE_FRAMES, RIGHT_FRAMES = 420, 50, 130
STATS_FRAMES = 3000
SIGMA_FLOOR = 1e-6
CHAR_ID = {c: i for i, c in enumerate(TOKENS)}


# ----------------------------------------------------------------------------------------------
# Features, exactly as mel.c / mel_tables.c / norm.c
# ----------------------------------------------------------------------------------------------
def _hz_to_mel(hz):
    f_sp, min_log_hz = 200.0 / 3.0, 1000.0
    min_log_mel, step = min_log_hz / f_sp, math.log(6.4) / 27.0
    return hz / f_sp if hz < min_log_hz else min_log_mel + math.log(hz / min_log_hz) / step


def _mel_to_hz(mel):
    f_sp, min_log_hz = 200.0 / 3.0, 1000.0
    min_log_mel, step = min_log_hz / f_sp, math.log(6.4) / 27.0
    return mel * f_sp if mel < min_log_mel else min_log_hz * math.exp(step * (mel - min_log_mel))


def mel_bank():
    """[64, 257] float32: the Slaney area-normalized triangles of mel_tables.c."""
    lo, hi = _hz_to_mel(0.0), _hz_to_mel(8000.0)
    edges = [_mel_to_hz(lo + (hi - lo) * i / (BANDS + 1)) for i in range(BANDS + 2)]
    step = 8000.0 / (NFFT // 2)
    bank = np.zeros((BANDS, NFFT // 2 + 1), np.float32)
    for b in range(BANDS):
        left, centre, right = edges[b], edges[b + 1], edges[b + 2]
        for k in range(NFFT // 2 + 1):
            f = k * step
            w = min((f - left) / (centre - left), (right - f) / (right - centre))
            if w > 0.0:
                bank[b, k] = np.float32(w * 2.0 / (right - left))
    return bank


def hann_window():
    n = np.arange(FRAME, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / (FRAME - 1))).astype(np.float32)


class LogMel(nn.Module):
    """Waveform (float, PCM16 / 32768) -> log-mel frames, as the board computes them."""

    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.from_numpy(hann_window()))
        self.register_buffer("bank", torch.from_numpy(mel_bank()))

    def forward(self, wave, lengths=None):
        """wave [B, N] float32 -> logmel [B, T, 64], frames [B] (valid frames per row)."""
        wave = wave.float()
        emph = torch.cat([wave[:, :1], wave[:, 1:] - PREEMPHASIS * wave[:, :-1]], dim=1)
        if emph.shape[1] < FRAME:
            emph = F.pad(emph, (0, FRAME - emph.shape[1]))
        frames = emph.unfold(1, FRAME, HOP) * self.window
        spec = torch.fft.rfft(frames, n=NFFT)
        power = spec.real.square() + spec.imag.square()
        logmel = torch.log(torch.matmul(power, self.bank.t()) + LOG_GUARD)
        if lengths is None:
            lengths = torch.full((wave.shape[0],), wave.shape[1], device=wave.device)
        n_frames = torch.clamp((lengths - FRAME) // HOP + 1, min=1)
        return logmel, n_frames


def logmel_numpy(pcm16):
    """One utterance (int16 numpy) -> log-mel [T, 64] numpy, via the torch code on the CPU."""
    lm = LogMel()
    wave = torch.from_numpy(pcm16.astype(np.float32) / 32768.0)[None]
    with torch.no_grad():
        x, n = lm(wave)
    return x[0, : int(n[0])].numpy()


def normalize_utterance(logmel, n_frames, jitter=0.0, mix=0.0):
    """Per-band mean/deviation over each utterance's own frames, then the board's int8 grid.

    The board uses the last 30 s of whatever it heard, several utterances and the room between
    them; per utterance is the training stand-in. With probability @p mix an utterance's statistics
    are blended (a random share) with those of the whole batch, i.e. of other speakers and rooms,
    and @p jitter adds a random error to them, so the model does not depend on them.
    Returns features [B, T, 64] with padded frames at 0, and the frame mask [B, T, 1].
    """
    B, T, _ = logmel.shape
    mask = (torch.arange(T, device=logmel.device)[None, :] < n_frames[:, None]).float()[..., None]
    count = mask.sum(1, keepdim=True).clamp(min=1.0)
    mean = (logmel * mask).sum(1, keepdim=True) / count
    var = ((logmel - mean).square() * mask).sum(1, keepdim=True) / count
    if mix > 0 and B > 1:
        total = mask.sum().clamp(min=1.0)
        b_mean = (logmel * mask).sum((0, 1), keepdim=True) / total
        b_var = ((logmel - b_mean).square() * mask).sum((0, 1), keepdim=True) / total
        share = (torch.rand(B, 1, 1, device=logmel.device) < mix).float() * torch.rand(B, 1, 1, device=logmel.device)
        # mixture of the two populations: its variance includes the spread between the means
        m = (1 - share) * mean + share * b_mean
        var = (1 - share) * (var + (mean - m).square()) + share * (b_var + (b_mean - m).square())
        mean = m
    std = var.clamp(min=0).sqrt().clamp(min=SIGMA_FLOOR)
    if jitter > 0:
        mean = mean + jitter * std * torch.randn_like(mean)
        std = std * torch.exp(jitter * torch.randn(B, 1, 1, device=logmel.device))
    x = (logmel - mean) / std
    x = torch.clamp(torch.round(x / INPUT_SCALE), -128, 127) * INPUT_SCALE
    return x * mask, mask


def board_windows(logmel):
    """norm.c over a whole stream: yields (frame_index, int8 window [600, 64]) every 50 frames.

    logmel [T, 64] float32 numpy. Frames before the stream are digital silence and stay out of
    the statistics, which cover the last 3000 frames in 50-frame blocks, as on the board.
    """
    silence = np.float32(np.log(np.float32(LOG_GUARD)))
    field = np.full((WIN_FRAMES, BANDS), silence, np.float32)
    n_blocks = STATS_FRAMES // STRIDE_FRAMES
    sums = np.zeros((n_blocks, BANDS)), np.zeros((n_blocks, BANDS))
    slot, filled, write = 0, 0, 0
    for t in range(0, logmel.shape[0] - STRIDE_FRAMES + 1, STRIDE_FRAMES):
        block = logmel[t: t + STRIDE_FRAMES].astype(np.float32)
        sums[0][slot] = block.astype(np.float64).sum(0)
        sums[1][slot] = np.square(block.astype(np.float64)).sum(0)
        for row in block:
            field[write] = row
            write = (write + 1) % WIN_FRAMES
        filled = min(filled + 1, n_blocks)
        frames = filled * STRIDE_FRAMES
        mean = sums[0][:filled].sum(0) / frames
        var = np.maximum(sums[1][:filled].sum(0) / frames - mean * mean, 0.0)
        dev = np.maximum(np.sqrt(var), SIGMA_FLOOR)
        slot = (slot + 1) % n_blocks
        ordered = np.concatenate([field[write:], field[:write]])
        norm = ((ordered.astype(np.float64) - mean) / dev).astype(np.float32)
        code = norm / np.float32(INPUT_SCALE) + np.float32(INPUT_ZP)
        code = np.clip(code, -128.0, 127.0)
        yield t + STRIDE_FRAMES, np.rint(code).astype(np.int8)  # rint: half to even, like lrintf


# ----------------------------------------------------------------------------------------------
# Text
# ----------------------------------------------------------------------------------------------
_PUNCT_MAP = {"!": ".", ";": ",", ":": ","}


def normalize_text(text, keep_punct):
    """Text -> the 32-class alphabet, or None when it cannot be said with it (digits, empty)."""
    if text is None:
        return None
    s = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").upper()
    if re.search(r"[0-9]", s):
        return None  # the alphabet has no digits; dropping beats guessing how they were read
    s = "".join(_PUNCT_MAP.get(c, c) for c in s)
    s = re.sub(r"[-_/–—]", " ", s)
    s = re.sub(r"[^A-Z' ,.?]", "", s)
    s = re.sub(r"(?<![A-Z])'|'(?![A-Z])", "", s)  # quotes, not apostrophes
    if not keep_punct:
        s = re.sub(r"[,.?]", " ", s)
    s = re.sub(r"\s+([,.?])", r"\1", s)  # marks stick to their word
    s = re.sub(r"([,.?])+", lambda m: max(m.group(0), key=lambda c: ",.?".index(c)), s)
    s = re.sub(r"([,.?])(?=[A-Z'])", r"\1 ", s)
    s = re.sub(r"^[ ,.?]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def text_to_ids(s):
    return [CHAR_ID[c] for c in s]


def words_of(s):
    """Words for the word error rate: marks removed."""
    return re.sub(r"[,.?]", " ", s).split()


def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[len(b)]


def greedy_text(ids):
    """CTC greedy decode of a step sequence of class ids."""
    out, prev = [], -1
    for i in ids:
        i = int(i)
        if i != prev and i != BLANK:
            out.append(TOKENS[i])
        prev = i
    return re.sub(r"\s+", " ", "".join(out)).strip()


# ----------------------------------------------------------------------------------------------
# The compiled network, from its ONNX file, as a trainable PyTorch module
# ----------------------------------------------------------------------------------------------
def _attrs(node):
    import onnx
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


class OnnxNet(nn.Module):
    """Runs an ONNX graph (Conv, Relu, Add, ... with QDQ) in PyTorch with trainable weights.

    Activation QuantizeLinear/DequantizeLinear pairs are skipped (float training); weights stored
    as int8 + DequantizeLinear become float parameters. forward() takes features [B, T, 64] and
    returns logits [B, T/2, 32] whatever layout the file uses.
    """

    SUPPORTED = {"Conv", "Relu", "Clip", "LeakyRelu", "Add", "Sub", "Mul", "Div", "Transpose", "Reshape",
                 "Squeeze", "Unsqueeze", "Identity", "Concat", "Pad", "BatchNormalization", "Sigmoid",
                 "Tanh", "HardSigmoid", "HardSwish", "Max", "Min"}

    def __init__(self, path):
        super().__init__()
        import onnx
        from onnx import numpy_helper
        self.path = path
        model = onnx.load(path)
        g = model.graph
        consts = {t.name: numpy_helper.to_array(t) for t in g.initializer}
        for n in g.node:
            if n.op_type == "Constant":
                consts[n.output[0]] = numpy_helper.to_array(_attrs(n)["value"])
        alias, nodes, quantizers = {}, [], {}
        self.weight_dq = {}  # float weight name -> (int8 name, scale name, zp name) in the file
        self.qdq_relu = set()  # DequantizeLinear outputs that stand for a ReLU
        for n in g.node:
            if n.op_type == "Constant":
                continue
            if n.op_type == "DequantizeLinear" and n.input[0] in consts:
                q = consts[n.input[0]].astype(np.float32)
                s = consts[n.input[1]].astype(np.float32)
                zp = consts[n.input[2]].astype(np.float32) if len(n.input) > 2 and n.input[2] else np.zeros_like(s)
                if s.ndim == 1 and s.size > 1:
                    axis = _attrs(n).get("axis", 1)
                    shape = [1] * q.ndim
                    shape[axis] = -1
                    s, zp = s.reshape(shape), zp.reshape(shape)
                consts[n.output[0]] = ((q - zp) * s).astype(np.float32)
                self.weight_dq[n.output[0]] = list(n.input)
                continue
            if n.op_type == "QuantizeLinear":
                quantizers[n.output[0]] = n
                alias[n.output[0]] = n.input[0]
                continue
            if n.op_type == "DequantizeLinear":
                # An activation's int8 step. When its range starts at 0 (zero point = the lowest
                # code), it is a ReLU: the file has no Relu nodes, the step cuts negatives off.
                q = quantizers.get(n.input[0])
                zp = consts.get(n.input[2]) if len(n.input) > 2 and n.input[2] else None
                lowest = (np.iinfo(zp.dtype).min if zp is not None else 0)  # no zero point: uint8
                if (q is not None) and np.all((np.zeros(1) if zp is None else np.asarray(zp)).astype(np.int64)
                                              == lowest):
                    relu = onnx.helper.make_node("Relu", [q.input[0]], [n.output[0]], name=n.output[0] + "_relu")
                    self.qdq_relu.add(n.output[0])
                    nodes.append(relu)
                else:
                    alias[n.output[0]] = n.input[0]
                continue
            if n.op_type not in self.SUPPORTED:
                raise ValueError("operator %s (%s) is not supported here" % (n.op_type, n.name))
            nodes.append(n)

        def resolve(name):
            while name in alias:
                name = alias[name]
            return name

        self.alias = {k: resolve(k) for k in alias}
        inits = {i.name for i in g.initializer}
        self.in_name = [i.name for i in g.input if i.name not in inits][0]
        self.out_name = resolve(g.output[0].name)
        self.graph_out = g.output[0].name
        in_dims = [d.dim_value for d in [i for i in g.input if i.name == self.in_name][0].type.tensor_type.shape.dim]
        out_dims = [d.dim_value for d in g.output[0].type.tensor_type.shape.dim]
        self.in_dims, self.out_dims = in_dims, out_dims
        self.in_time, self.in_band = in_dims.index(WIN_FRAMES), in_dims.index(BANDS)
        self.out_time, self.out_class = out_dims.index(WIN_STEPS), out_dims.index(len(TOKENS))

        self.params = nn.ParameterDict()
        self.pname = {}  # tensor name in the file -> key in self.params
        self.ops = []
        for n in nodes:
            ins = [resolve(x) if x else "" for x in n.input]
            if n.op_type in ("Conv", "BatchNormalization"):
                for x in ins[1:]:
                    if x and x not in self.pname:
                        key = "p%d" % len(self.pname)
                        self.pname[x] = key
                        self.params[key] = nn.Parameter(torch.from_numpy(np.array(consts[x], np.float32)))
            self.ops.append((n.op_type, ins, list(n.output), _attrs(n), n.name))
        self.consts = {k: torch.from_numpy(np.array(v)) for k, v in consts.items()
                       if k not in self.pname and np.asarray(v).dtype != object}

    # -- layout helpers
    def to_graph_input(self, x):
        """[B, T, 64] -> the file's input layout."""
        roles = {0: "b", self.in_time: "t", self.in_band: "f"}
        seq = [roles[a] for a in range(len(self.in_dims)) if a in roles]
        y = x.permute([["b", "t", "f"].index(r) for r in seq])
        for a in range(len(self.in_dims)):
            if a not in roles:
                y = y.unsqueeze(a)
        return y

    def from_graph_output(self, y):
        """The file's output layout -> [B, T', 32]."""
        roles = {0: "b", self.out_time: "t", self.out_class: "c"}
        for a in reversed(range(len(self.out_dims))):
            if a not in roles:
                y = y.squeeze(a)
        seq = [roles[a] for a in range(len(self.out_dims)) if a in roles]
        return y.permute([seq.index(r) for r in ["b", "t", "c"]])

    def _value(self, env, name, like):
        if name in env:
            return env[name]
        if name in self.pname:
            return self.params[self.pname[name]]
        t = self.consts[name]
        return t.to(like.device, like.dtype) if t.is_floating_point() else t.to(like.device)

    @staticmethod
    def _conv(x, w, b, a):
        k = w.dim() - 2
        strides, dil = a.get("strides", [1] * k), a.get("dilations", [1] * k)
        group = a.get("group", 1)
        pads = list(a.get("pads", [0] * 2 * k))
        auto = a.get("auto_pad", b"NOTSET")
        auto = auto.decode() if isinstance(auto, bytes) else auto
        if auto in ("SAME_UPPER", "SAME_LOWER"):
            pads = [0] * 2 * k
            for i in range(k):
                L = x.shape[2 + i]
                ks = (w.shape[2 + i] - 1) * dil[i] + 1
                total = max((math.ceil(L / strides[i]) - 1) * strides[i] + ks - L, 0)
                small, big = total // 2, total - total // 2
                pads[i], pads[i + k] = (small, big) if auto == "SAME_UPPER" else (big, small)
        elif auto == "VALID":
            pads = [0] * 2 * k
        conv = F.conv1d if k == 1 else F.conv2d
        if pads[:k] == pads[k:]:
            return conv(x, w, b, strides, pads[:k], dil, group)
        torch_pad = []
        for i in reversed(range(k)):
            torch_pad += [pads[i], pads[i + k]]
        return conv(F.pad(x, torch_pad), w, b, strides, 0, dil, group)

    def forward(self, feats):
        x = self.to_graph_input(feats)
        env = {self.in_name: x}
        batch = x.shape[0]
        for op, ins, outs, a, _ in self.ops:
            v = [self._value(env, n, x) if n else None for n in ins]
            if op == "Conv":
                y = self._conv(v[0], v[1], v[2] if len(v) > 2 else None, a)
            elif op == "Relu":
                y = F.relu(v[0])
            elif op == "Clip":
                lo = v[1] if len(v) > 1 and v[1] is not None else a.get("min")
                hi = v[2] if len(v) > 2 and v[2] is not None else a.get("max")
                y = torch.clamp(v[0], None if lo is None else float(lo), None if hi is None else float(hi))
            elif op == "LeakyRelu":
                y = F.leaky_relu(v[0], a.get("alpha", 0.01))
            elif op == "Sigmoid":
                y = torch.sigmoid(v[0])
            elif op == "Tanh":
                y = torch.tanh(v[0])
            elif op == "HardSigmoid":
                y = torch.clamp(a.get("alpha", 0.2) * v[0] + a.get("beta", 0.5), 0, 1)
            elif op == "HardSwish":
                y = F.hardswish(v[0])
            elif op in ("Add", "Sub", "Mul", "Div", "Max", "Min"):
                fn = {"Add": torch.add, "Sub": torch.sub, "Mul": torch.mul, "Div": torch.div,
                      "Max": torch.maximum, "Min": torch.minimum}[op]
                y = fn(v[0], v[1])
            elif op == "Transpose":
                y = v[0].permute(a["perm"])
            elif op == "Reshape":
                shape = [int(s) for s in v[1].tolist()]
                if shape and shape[0] == 1:
                    shape[0] = batch
                for i, s in enumerate(shape):
                    if s in (WIN_FRAMES, WIN_STEPS) and -1 not in shape:
                        shape[i] = -1
                y = v[0].reshape(shape)
            elif op == "Squeeze":
                axes = a.get("axes", v[1].tolist() if len(v) > 1 and v[1] is not None else None)
                y = v[0]
                for ax in sorted([ax % v[0].dim() for ax in axes], reverse=True) if axes is not None else []:
                    y = y.squeeze(ax)
            elif op == "Unsqueeze":
                axes = a.get("axes", v[1].tolist() if len(v) > 1 and v[1] is not None else None)
                y = v[0]
                for ax in sorted(axes):
                    y = y.unsqueeze(ax)
            elif op == "Identity":
                y = v[0]
            elif op == "Concat":
                y = torch.cat(v, dim=a["axis"])
            elif op == "Pad":
                pads = [int(p) for p in (v[1].tolist() if len(v) > 1 else a["pads"])]
                k = len(pads) // 2
                torch_pad = []
                for i in reversed(range(k)):
                    torch_pad += [pads[i], pads[i + k]]
                y = F.pad(v[0], torch_pad)
            elif op == "BatchNormalization":
                eps = a.get("epsilon", 1e-5)
                shape = [1, -1] + [1] * (v[0].dim() - 2)
                y = (v[0] - v[3].reshape(shape)) / torch.sqrt(v[4].reshape(shape) + eps) * v[1].reshape(shape) \
                    + v[2].reshape(shape)
            else:
                raise ValueError(op)
            env[outs[0]] = y
        return self.from_graph_output(env[self.out_name])

    def time_stride(self):
        return WIN_FRAMES // WIN_STEPS

    def export_float_onnx(self, dst):
        """The original graph without any Q/DQ node, with this module's weights."""
        import onnx
        from onnx import helper, numpy_helper
        model = onnx.load(self.path)
        g = model.graph
        keep = []
        for n in g.node:
            if n.op_type == "DequantizeLinear" and n.output[0] in self.qdq_relu:
                source = self.alias.get(n.input[0], n.input[0])
                keep.append(helper.make_node("Relu", [source], [n.output[0]], name=n.output[0] + "_relu"))
                continue
            if n.op_type in ("QuantizeLinear", "DequantizeLinear"):
                continue
            if n.op_type == "Constant":
                keep.append(n)
                continue
            for i, x in enumerate(n.input):
                if x in self.alias:
                    n.input[i] = self.alias[x]
            keep.append(n)
        if self.graph_out != self.out_name:
            keep.append(helper.make_node("Identity", [self.out_name], [self.graph_out], name="output_identity"))
        del g.node[:]
        g.node.extend(keep)
        trained = {name: self.params[key].detach().float().cpu().numpy() for name, key in self.pname.items()}
        kept_inits = [t for t in g.initializer if t.name not in trained]
        del g.initializer[:]
        g.initializer.extend(kept_inits)
        g.initializer.extend(numpy_helper.from_array(v, name) for name, v in trained.items())
        used = {x for n in g.node for x in n.input}
        alive = [t for t in g.initializer if t.name in used]
        del g.initializer[:]
        g.initializer.extend(alive)
        consts = [n for n in g.node if n.op_type != "Constant" or n.output[0] in used]
        del g.node[:]
        g.node.extend(consts)
        del g.value_info[:]
        onnx.checker.check_model(model)
        onnx.save(model, dst)
        return dst


def count_params(net):
    return sum(p.numel() for p in net.parameters())


# ----------------------------------------------------------------------------------------------
# A CynipsNet-shaped network from the compile report: only for the PC smoke test (and as a
# documented last resort); real training starts from the board's ONNX file.
# ----------------------------------------------------------------------------------------------
class _Sep(nn.Module):
    def __init__(self, cin, cout, k, stride=1):
        super().__init__()
        self.dw = nn.Conv1d(cin, cin, k, stride, k // 2, groups=cin)
        self.pw = nn.Conv1d(cin, cout, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


class CynipsLike(nn.Module):
    """Input [1, 64, T], output [1, 32, T/2]; 7.2 M parameters like the compile report."""

    BLOCKS = [(256, 256, 33), (256, 256, 39), (256, 256, 39), (256, 512, 51), (512, 512, 63), (512, 512, 75)]

    def __init__(self, classes=len(TOKENS)):
        super().__init__()
        self.c1 = _Sep(BANDS, 256, 33, 2)
        self.blocks = nn.ModuleList()
        for cin, cout, k in self.BLOCKS:
            subs = nn.ModuleList([_Sep(cin if i == 0 else cout, cout, k) for i in range(5)])
            self.blocks.append(nn.ModuleDict({"subs": subs, "res": nn.Conv1d(cin, cout, 1)}))
        self.c2 = _Sep(512, 512, 173)
        self.c3 = nn.Conv1d(512, 1024, 1)
        self.c4 = nn.Conv1d(1024, classes, 1)

    def forward(self, x):
        x = F.relu(self.c1(x))
        for b in self.blocks:
            y = x
            for i, s in enumerate(b["subs"]):
                y = s(y)
                if i < 4:
                    y = F.relu(y)
            x = F.relu(y + b["res"](x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        return self.c4(x)


# ----------------------------------------------------------------------------------------------
# Data: Hugging Face parquet shards on local disk, read one row group at a time
# ----------------------------------------------------------------------------------------------
@dataclass
class Source:
    name: str
    repo: str
    patterns: list
    text_col: str
    punct: bool                 # transcripts carry , . ? the board model must learn
    shards: int = 0             # 0 = every matching file; else this many, spread over the list
    replay: bool = False        # old-domain data, kept to stop forgetting (teacher KL applies)
    hours_per_epoch: float = 0  # replay only: hours drawn per epoch (0 = all)
    audio_col: str = "audio"
    dur_col: str = ""           # column with the duration, if any
    dur_scale: float = 1.0      # e.g. 0.001 for milliseconds
    speaker_col: str = ""       # used to hold out dev speakers
    filter_col: str = ""
    filter_has: str = ""
    lspc: bool = False          # take punctuated text from LibriSpeech-PC by utterance id
    id_col: str = ""
    gated: bool = False


def select_files(files, src):
    files = sorted(f for f in files if any(fnmatch(f, p) for p in src.patterns))
    if src.shards and len(files) > src.shards:
        step = len(files) / src.shards
        files = [files[int(i * step)] for i in range(src.shards)]
    return files


def download(src, root, token=None):
    """The selected parquet files of one source into root/<name>; returns their local paths."""
    from huggingface_hub import HfApi, snapshot_download
    files = select_files(HfApi(token=token).list_repo_files(src.repo, repo_type="dataset"), src)
    local = os.path.join(root, src.name)
    snapshot_download(src.repo, repo_type="dataset", allow_patterns=files, local_dir=local, token=token,
                      max_workers=8)
    return [os.path.join(local, f) for f in files]


def decode_audio(cell):
    """HF Audio cell -> mono int16 numpy at 16 kHz."""
    import soundfile as sf
    from scipy.signal import resample_poly
    data = cell["bytes"] if isinstance(cell, dict) else cell
    if data is None and isinstance(cell, dict) and cell.get("array") is not None:
        x, rate = np.asarray(cell["array"], np.float32), cell["sampling_rate"]
    else:
        x, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        x = x.mean(1)
    if rate != SR:
        g = math.gcd(int(rate), SR)
        x = resample_poly(x, SR // g, int(rate) // g).astype(np.float32)
    return np.clip(np.round(x * 32768.0), -32768, 32767).astype(np.int16)


def audio_seconds(cell):
    import soundfile as sf
    try:
        return sf.info(io.BytesIO(cell["bytes"])).duration
    except Exception:
        return -1.0


def load_lspc(path):
    """LibriSpeech-PC manifests (openslr.org/145) -> {utterance id: punctuated text}."""
    import tarfile
    texts = {}
    with tarfile.open(path) as tar:
        for m in tar.getmembers():
            if m.name.endswith(".json"):
                for line in tar.extractfile(m):
                    r = json.loads(line)
                    texts[os.path.basename(r["audio_filepath"]).rsplit(".", 1)[0]] = r["text"]
    return texts


def _speaker_is_dev(speaker):
    return speaker is not None and (hash_str(str(speaker)) % 50 == 0)


def hash_str(s):
    h = 2166136261
    for c in s.encode():
        h = ((h ^ c) * 16777619) & 0xFFFFFFFF
    return h


def index_file(args):
    """One parquet file -> list of units: (row_group, hours, rows kept). Runs in a pool."""
    import pyarrow.parquet as pq
    path, src, lspc_path = args
    lspc = _LSPC_CACHE.get(lspc_path) if src.lspc else None
    if src.lspc and lspc is None and lspc_path:
        lspc = _LSPC_CACHE.setdefault(lspc_path, load_lspc(lspc_path))
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    cols = [c for c in (src.text_col, src.dur_col, src.speaker_col, src.filter_col, src.id_col) if c and c in names]
    need_audio = not (src.dur_col and src.dur_col in names)
    units = []
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=cols + ([src.audio_col] if need_audio else [])).to_pylist()
        rows, secs = [], 0.0
        for i, r in enumerate(t):
            if src.filter_col and src.filter_has not in str(r.get(src.filter_col, "")):
                continue
            if src.speaker_col and _speaker_is_dev(r.get(src.speaker_col)):
                continue
            text = r.get(src.text_col)
            punct = src.punct
            if lspc is not None:
                pc = lspc.get(str(r.get(src.id_col)))
                if pc is not None:
                    text, punct = pc, True
            norm = normalize_text(text, punct)
            if norm is None:
                continue
            d = float(r[src.dur_col]) * src.dur_scale if not need_audio else audio_seconds(r[src.audio_col])
            if not (0.6 <= d <= 20.0):
                continue
            if len(norm) > d * 25:  # more characters than the 50 steps per second can carry
                continue
            rows.append((i, norm, punct))
            secs += d
        if rows:
            units.append((path, rg, secs / 3600.0, rows))
    return units


_LSPC_CACHE = {}


def build_index(src, files, cache_path, lspc_path=None, workers=None):
    """Units of one source, cached as JSON on Drive so later sessions skip the scan."""
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        root = os.path.dirname(files[0]) if files else ""
        by_name = {os.path.basename(p): p for p in files}
        units = [(by_name.get(os.path.basename(u[0]), u[0]), u[1], u[2], [tuple(r) for r in u[3]]) for u in cached]
        units = [u for u in units if os.path.exists(u[0])]
        if units:
            return units
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as pool:
        parts = list(pool.map(index_file, [(p, src, lspc_path) for p in files]))
    units = [u for part in parts for u in part]
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(units, f)
    return units


def dev_set(src, files, n, lspc=None, split_speakers=True):
    """Up to n utterances held out for evaluation: [(pcm16, text)]."""
    import pyarrow.parquet as pq
    out = []
    for path in files:
        pf = pq.ParquetFile(path)
        names = pf.schema_arrow.names
        for rg in range(pf.num_row_groups):
            for r in pf.read_row_group(rg).to_pylist():
                if src.filter_col and src.filter_has not in str(r.get(src.filter_col, "")):
                    continue
                if split_speakers and src.speaker_col and not _speaker_is_dev(r.get(src.speaker_col)):
                    continue
                text, punct = r.get(src.text_col), src.punct
                if lspc is not None and src.id_col:
                    pc = lspc.get(str(r.get(src.id_col)))
                    if pc is not None:
                        text, punct = pc, True
                norm = normalize_text(text, True)
                if norm is None:
                    continue
                pcm = decode_audio(r[src.audio_col])
                if not (0.6 * SR <= len(pcm) <= 20 * SR):
                    continue
                out.append((pcm, norm))
                if len(out) >= n:
                    return out
    return out


# -- augmentation (numpy, in the loader workers)
def _colored_noise(n, rng):
    if n < 2:
        return np.zeros(max(int(n), 0), np.float32)
    white = rng.standard_normal(n).astype(np.float32)
    kind = rng.integers(3)
    if kind == 0:
        return white
    spec = np.fft.rfft(white)
    f = np.arange(spec.size, dtype=np.float32) + 1.0
    spec /= np.sqrt(f) if kind == 1 else f
    x = np.fft.irfft(spec, n).astype(np.float32)
    return x / (x.std() + 1e-9)


def augment(x, rng, babble=None):
    """int16 utterance -> augmented int16: speed, room, band limit, noise, level."""
    from scipy.signal import fftconvolve, lfilter, resample_poly, butter
    y = x.astype(np.float32) / 32768.0
    r = rng.random()
    if r < 0.25:
        y = resample_poly(y, 10, 9).astype(np.float32)   # slower (0.9x)
    elif r < 0.5:
        y = resample_poly(y, 10, 11).astype(np.float32)  # faster (1.1x)
    if rng.random() < 0.3:  # a room: exponentially decaying noise as the impulse response
        rt60 = rng.uniform(0.15, 0.7)
        n = int(rt60 * SR)
        rir = rng.standard_normal(n).astype(np.float32) * np.exp(-6.9 * np.arange(n) / n).astype(np.float32)
        rir[0] = 1.0
        y = fftconvolve(y, rir / np.sqrt(np.sum(rir ** 2)))[: len(y)].astype(np.float32)
    if rng.random() < 0.4:  # the board microphone and room: band limits
        lo = rng.uniform(60, 250)
        hi = rng.uniform(3800, 7600)
        b, a = butter(2, [lo / (SR / 2), hi / (SR / 2)], btype="band")
        y = lfilter(b, a, y).astype(np.float32)
    level = np.sqrt(np.mean(y ** 2)) + 1e-9
    if rng.random() < 0.5:
        snr = rng.uniform(3, 30)
        if babble is not None and rng.random() < 0.3 and len(babble) > 0:
            b = babble[rng.integers(len(babble))].astype(np.float32) / 32768.0
            reps = int(np.ceil(len(y) / max(len(b), 1)))
            noise = np.tile(b, reps)[: len(y)]
            noise = noise / (np.sqrt(np.mean(noise ** 2)) + 1e-9)
        else:
            noise = _colored_noise(len(y), rng)
        y = y + noise * level / (10 ** (snr / 20))
    lead, tail = rng.integers(0, int(0.4 * SR)), rng.integers(0, int(0.4 * SR))
    floor = rng.uniform(1e-4, 3e-3) * level
    y = np.concatenate([_colored_noise(lead, rng) * floor, y, _colored_noise(tail, rng) * floor])
    gain = 10 ** (rng.uniform(-26, -3) / 20) / (np.max(np.abs(y)) + 1e-9)  # peak at -26..-3 dBFS
    return np.clip(np.round(y * gain * 32768.0), -32768, 32767).astype(np.int16)


class ShardStream(torch.utils.data.IterableDataset):
    """Batches of (pcm16 padded, lengths, targets, target lengths, punct flags, replay flags).

    Each epoch the row groups of every source are shuffled and split over the workers; rows go
    into a pool that is cut into batches of similar length, at most batch_seconds of padded audio.
    """

    def __init__(self, units_by_source, sources, epoch, batch_seconds=320.0, max_batch=64, pool=1500,
                 augment_p=0.8, seed=1234):
        super().__init__()
        self.units_by_source, self.sources = units_by_source, {s.name: s for s in sources}
        self.epoch, self.batch_seconds, self.max_batch = epoch, batch_seconds, max_batch
        self.pool, self.augment_p, self.seed = pool, augment_p, seed

    def epoch_units(self):
        rng = random.Random(self.seed * 1000 + self.epoch)
        chosen = []
        for name, units in self.units_by_source.items():
            src = self.sources[name]
            units = list(units)
            rng.shuffle(units)
            if src.replay and src.hours_per_epoch > 0:
                take, hours = [], 0.0
                for u in units:
                    if hours >= src.hours_per_epoch:
                        break
                    take.append(u)
                    hours += u[2]
                units = take
            chosen += [(name, u) for u in units]
        rng.shuffle(chosen)
        return chosen

    def hours(self):
        return sum(u[2] for _, u in self.epoch_units())

    def __iter__(self):
        import pyarrow.parquet as pq
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        units = self.epoch_units()[wid::nw]
        rng = np.random.default_rng(self.seed + 7919 * self.epoch + wid)
        pool, babble = [], []
        files = {}
        for name, (path, rg, _, rows) in units:
            src = self.sources[name]
            pf = files.get(path) or files.setdefault(path, pq.ParquetFile(path))
            col = pf.read_row_group(rg, columns=[src.audio_col]).column(0).to_pylist()
            for i, text, punct in rows:
                try:
                    pcm = decode_audio(col[i])
                except Exception:
                    continue
                if len(babble) < 32 and rng.random() < 0.05:
                    babble.append(pcm)
                if rng.random() < self.augment_p:
                    pcm = augment(pcm, rng, babble)
                pool.append((pcm, text_to_ids(text), punct, src.replay))
            if len(pool) >= self.pool:
                yield from self._batches(pool, rng)
                pool = []
            if len(files) > 8:
                files.clear()
        if pool:
            yield from self._batches(pool, rng)

    def _batches(self, pool, rng):
        pool.sort(key=lambda s: len(s[0]))
        batches, cur = [], []
        for s in pool:
            longest = max(len(s[0]), max((len(c[0]) for c in cur), default=0))
            if cur and ((len(cur) + 1) * longest > self.batch_seconds * SR or len(cur) >= self.max_batch):
                batches.append(cur)
                cur = []
            cur.append(s)
        if cur:
            batches.append(cur)
        order = rng.permutation(len(batches))
        for k in order:
            yield collate(batches[k])


def collate(samples):
    n = max(len(s[0]) for s in samples)
    wave = torch.zeros(len(samples), n, dtype=torch.int16)
    lengths = torch.zeros(len(samples), dtype=torch.long)
    tl = torch.tensor([len(s[1]) for s in samples], dtype=torch.long)
    targets = torch.zeros(len(samples), int(tl.max()), dtype=torch.long)
    for i, s in enumerate(samples):
        wave[i, : len(s[0])] = torch.from_numpy(s[0])
        lengths[i] = len(s[0])
        targets[i, : len(s[1])] = torch.tensor(s[1])
    punct = torch.tensor([s[2] for s in samples], dtype=torch.bool)
    replay = torch.tensor([s[3] for s in samples], dtype=torch.bool)
    return wave, lengths, targets, tl, punct, replay


# ----------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------
def spec_augment(x, mask, rng_gen=None, freq_masks=2, freq_width=8, time_masks=5, time_frac=0.04):
    """Zero (the normalized mean) random bands and spans, in place on a copy."""
    x = x.clone()
    B, T, Fb = x.shape
    for b in range(B):
        valid = int(mask[b, :, 0].sum())
        for _ in range(freq_masks):
            w = int(torch.randint(0, freq_width + 1, (1,)))
            f0 = int(torch.randint(0, max(1, Fb - w), (1,)))
            x[b, :, f0: f0 + w] = 0
        for _ in range(time_masks):
            w = int(torch.randint(0, max(1, int(valid * time_frac)) + 1, (1,)))
            t0 = int(torch.randint(0, max(1, valid - w), (1,)))
            x[b, t0: t0 + w, :] = 0
    return x


@dataclass
class TrainConfig:
    lr: float = 2e-4
    weight_decay: float = 1e-3
    warmup_steps: int = 800
    epochs: int = 20
    batch_seconds: float = 480.0     # padded audio per batch; lower it if the GPU runs out of memory
    max_batch: int = 64
    kl_weight: float = 0.5          # keep the old behaviour on replay (LibriSpeech) samples
    norm_jitter: float = 0.1
    norm_mix: float = 0.5           # share of rows normalized partly with other rows' statistics
    spec_augment: bool = True
    grad_clip: float = 1.0
    amp: bool = True
    eval_every: int = 2000
    log_every: int = 100
    save_minutes: float = 15.0
    session_hours: float = 11.0     # stop and save before Colab ends the session
    workers: int = 0                # 0 = number of CPUs
    seed: int = 1234


class Trainer:
    def __init__(self, onnx_path, out_dir, cfg, device):
        self.cfg, self.device, self.out_dir = cfg, device, out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.model = OnnxNet(onnx_path).to(device)
        self.teacher = None
        if cfg.kl_weight > 0:
            self.teacher = OnnxNet(onnx_path).to(device).eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)
        self.logmel = LogMel().to(device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
        self.step, self.epoch, self.best = 0, 0, float("inf")
        self.nonfinite = 0
        self.total_steps = 1
        self.audio_hours = 0.0
        self.history = []

    # -- checkpoints (on Drive: survive a disconnect)
    def save(self, name):
        path = os.path.join(self.out_dir, name)
        torch.save({"model": self.model.state_dict(), "opt": self.opt.state_dict(),
                    "scaler": self.scaler.state_dict(), "step": self.step, "epoch": self.epoch,
                    "best": self.best, "total_steps": self.total_steps, "audio_hours": self.audio_hours,
                    "history": self.history}, path + ".tmp")
        os.replace(path + ".tmp", path)

    def load(self, name="last.pt"):
        path = os.path.join(self.out_dir, name)
        if not os.path.exists(path):
            return False
        ck = torch.load(path, map_location=self.device, weights_only=False)
        if not all(bool(torch.isfinite(v).all()) for v in ck["model"].values() if torch.is_tensor(v)):
            print("%s holds weights that are not numbers: not resumed, starting again" % path)
            return False
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        self.scaler.load_state_dict(ck["scaler"])
        self.step, self.epoch, self.best = ck["step"], ck["epoch"], ck["best"]
        self.total_steps, self.audio_hours = ck["total_steps"], ck["audio_hours"]
        self.history = ck.get("history", [])
        return True

    def lr_at(self, step):
        c = self.cfg
        if step < c.warmup_steps:
            return c.lr * (step + 1) / c.warmup_steps
        p = min(1.0, (step - c.warmup_steps) / max(1, self.total_steps - c.warmup_steps))
        return 1e-6 + (c.lr - 1e-6) * 0.5 * (1 + math.cos(math.pi * p))

    def features(self, wave, lengths, train):
        logmel, frames = self.logmel(wave.to(self.device).float() / 32768.0, lengths.to(self.device))
        x, mask = normalize_utterance(logmel, frames, self.cfg.norm_jitter if train else 0.0,
                                      self.cfg.norm_mix if train else 0.0)
        if train and self.cfg.spec_augment:
            x = spec_augment(x, mask)
        return x, frames

    def train_step(self, batch):
        wave, lengths, targets, tl, punct, replay = batch
        x, frames = self.features(wave, lengths, True)
        dev = self.device
        with torch.autocast(dev.type, dtype=torch.float16, enabled=self.cfg.amp and dev.type == "cuda"):
            logits = self.model(x)
            sel = replay.to(dev)
            t_logits = self.teacher(x[sel]) if (self.teacher is not None and bool(replay.any())) else None
        logits = logits.float()
        stride = self.model.time_stride()
        out_len = torch.clamp((frames + stride - 1) // stride, max=logits.shape[1])
        masked = logits.clone()
        no_punct = (~punct).to(dev)
        if bool(no_punct.any()):  # no marks in these transcripts: leave the mark classes out
            masked[no_punct, :, PUNCT_FIRST:] = -1e4
        logp = F.log_softmax(masked, -1)
        ctc = F.ctc_loss(logp.transpose(0, 1), targets.to(dev), out_len, tl.to(dev), blank=BLANK,
                         reduction="mean", zero_infinity=True)
        loss, kl = ctc, torch.zeros((), device=dev)
        if t_logits is not None:  # the old model's outputs on old-domain rows: stay close to them
            t = F.log_softmax(t_logits.float(), -1)
            s = F.log_softmax(logits[sel], -1)
            m = (torch.arange(logits.shape[1], device=dev)[None, :] < out_len[sel][:, None]).float()
            kl = ((t.exp() * (t - s)).sum(-1) * m).sum() / m.sum().clamp(min=1)
            loss = loss + self.cfg.kl_weight * kl
        if not bool(torch.isfinite(loss)):
            # nothing to learn from this batch (an overflow): skip it, and give up if it keeps happening
            self.nonfinite += 1
            if self.nonfinite >= 50:
                raise RuntimeError("50 batches in a row gave a loss that is not a number: training stopped. "
                                   "Send the output of cells 4 and 8.")
            return float("nan"), float("nan")
        self.nonfinite = 0
        for g in self.opt.param_groups:
            g["lr"] = self.lr_at(self.step)
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.step += 1
        self.audio_hours += float(lengths.sum()) / SR / 3600.0
        return float(ctc.detach()), float(kl.detach())

    @torch.no_grad()
    def transcribe(self, pcms, model=None, batch_seconds=200.0):
        """Whole utterances, greedy: the fast check used during training."""
        model = model or self.model
        model.eval()
        order = sorted(range(len(pcms)), key=lambda i: len(pcms[i]))
        texts = [None] * len(pcms)
        i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and (j - i + 1) * len(pcms[order[j]]) <= batch_seconds * SR:
                j += 1
            idx = order[i:j]
            wave, lengths, _, _, _, _ = collate([(pcms[k], [1], False, False) for k in idx])
            x, frames = self.features(wave, lengths, False)
            logits = model(x).float()  # full precision: the check must not depend on fp16 ranges
            stride = model.time_stride()
            for n, k in enumerate(idx):
                steps = int((frames[n] + stride - 1) // stride)
                texts[k] = greedy_text(logits[n, :steps].argmax(-1).tolist())
            i = j
        model.train()
        return texts

    def evaluate(self, dev_sets, model=None):
        res = {}
        for name, items in dev_sets.items():
            if not items:
                continue
            hyp = self.transcribe([p for p, _ in items], model)
            res[name] = wer([t for _, t in items], hyp)
        return res

    def fit(self, units_by_source, sources, dev_sets, log=print):
        cfg = self.cfg
        t_start, t_save = time.time(), time.time()
        workers = cfg.workers or os.cpu_count() or 2
        est_hours = ShardStream(units_by_source, sources, 0).hours()
        est_steps_per_epoch = max(1, int(est_hours * 3600 / (cfg.batch_seconds * 0.8)))
        self.total_steps = max(self.total_steps, est_steps_per_epoch * cfg.epochs)
        log("about %.0f h of audio per epoch, about %d steps per epoch, %d epochs" %
            (est_hours, est_steps_per_epoch, cfg.epochs))
        self.model.train()
        stop = False
        while self.epoch < cfg.epochs and not stop:
            ds = ShardStream(units_by_source, sources, self.epoch, cfg.batch_seconds, cfg.max_batch,
                             seed=cfg.seed)
            loader = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=workers,
                                                 prefetch_factor=4 if workers else None,
                                                 persistent_workers=False, pin_memory=True)
            t_log, h_log, sums = time.time(), self.audio_hours, [0.0, 0.0, 0]
            for batch in loader:
                ctc, kl = self.train_step(batch)
                if math.isnan(ctc):
                    continue
                sums[0] += ctc
                sums[1] += kl
                sums[2] += 1
                if self.step % cfg.log_every == 0:
                    dt = time.time() - t_log
                    speed = (self.audio_hours - h_log) * 3600 / max(dt, 1e-6)
                    log("epoch %d step %d  ctc %.3f  kl %.3f  lr %.2e  %.0fx real time  %.1f h of audio so far"
                        % (self.epoch, self.step, sums[0] / sums[2], sums[1] / sums[2], self.lr_at(self.step),
                           speed, self.audio_hours))
                    t_log, h_log, sums = time.time(), self.audio_hours, [0.0, 0.0, 0]
                if self.step % cfg.eval_every == 0:
                    self.checkpoint_eval(dev_sets, log)
                if time.time() - t_save > cfg.save_minutes * 60:
                    self.save("last.pt")
                    t_save = time.time()
                if time.time() - t_start > cfg.session_hours * 3600:
                    log("session time used up: saving; run the training cell again to continue")
                    stop = True
                    break
            if not stop:
                self.epoch += 1
                self.checkpoint_eval(dev_sets, log)
            self.save("last.pt")
        return self.best

    def checkpoint_eval(self, dev_sets, log):
        res = self.evaluate(dev_sets)
        indian = [v["wer"] for k, v in res.items() if not k.startswith("libri")]
        score = float(np.mean(indian)) if indian else float(np.mean([v["wer"] for v in res.values()]))
        self.history.append({"step": self.step, "epoch": self.epoch, **{k: v["wer"] for k, v in res.items()}})
        log("  eval step %d: " % self.step + "  ".join("%s WER %.2f%%" % (k, 100 * v["wer"]) for k, v in res.items()))
        if score < self.best:
            self.best = score
            self.save("best.pt")
            log("  new best (Indian dev WER %.2f%%), saved best.pt" % (100 * score))


def wer(refs, hyps):
    errors = words = 0
    for r, h in zip(refs, hyps):
        rw, hw = words_of(r), words_of(h or "")
        errors += edit_distance(rw, hw)
        words += len(rw)
    return {"wer": errors / max(1, words), "words": words, "n": len(refs)}


# ----------------------------------------------------------------------------------------------
# Export: float ONNX -> int8 QDQ (onnxruntime), the board's input scale, network_shape.h
# ----------------------------------------------------------------------------------------------
def graph_layout(path):
    import onnx
    g = onnx.load(path).graph
    inits = {i.name for i in g.initializer}
    inp = [i for i in g.input if i.name not in inits][0]
    in_dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    out_dims = [d.dim_value for d in g.output[0].type.tensor_type.shape.dim]
    return inp.name, in_dims, out_dims


def window_to_input(window_codes, in_dims):
    """int8 codes [600, 64] -> the float input the QDQ graph quantizes back to the same codes."""
    x = window_codes.astype(np.float32) * np.float32(INPUT_SCALE)
    t, f = in_dims.index(WIN_FRAMES), in_dims.index(BANDS)
    x = x if t < f else x.T
    shape = [1] * len(in_dims)
    shape[t], shape[f] = WIN_FRAMES, BANDS
    return x.reshape(shape)


def output_to_steps(y, out_dims):
    t, c = out_dims.index(WIN_STEPS), out_dims.index(len(TOKENS))
    y = np.squeeze(y)
    return y if t < c else y.T


def calibration_windows(dev_items, count=300, seed=0):
    rng = random.Random(seed)
    wins = []
    for pcm, _ in dev_items:
        for _, w in board_windows(logmel_numpy(pcm)):
            wins.append(w)
    rng.shuffle(wins)
    return wins[:count]


def quantize_int8(float_path, qdq_path, windows, method="minmax"):
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    name, in_dims, _ = graph_layout(float_path)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(windows)

        def get_next(self):
            w = next(self.it, None)
            return None if w is None else {name: window_to_input(w, in_dims)}

    quantize_static(float_path, qdq_path, Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.Percentile if method == "percentile" else CalibrationMethod.MinMax,
                    extra_options={"ActivationSymmetric": False, "WeightSymmetric": True})
    return finish_qdq(qdq_path, windows)


def finish_qdq(qdq_path, windows):
    """Make the int8 graph's ends what the board expects; returns (input scale, output scale).

    - Input: a QuantizeLinear/DequantizeLinear pair right on the graph input with the board's
      scale and zero point (norm.c writes codes with them). Inserted if the quantizer put the
      first pair after a layout operator.
    - Output: a pair on the graph output with zero point 0 (the board reads int8 logits; ST Edge
      AI needs the scale for --output-data-type int8). Added when onnxruntime left the output in
      float (the largest output over the calibration windows maps to 127), made symmetric when it
      quantized it with another zero point.
    """
    import onnx
    import onnxruntime as ort
    from onnx import helper, numpy_helper
    m = onnx.load(qdq_path)
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    name = [i.name for i in g.input if i.name not in inits][0]

    q = [n for n in g.node if n.op_type == "QuantizeLinear" and n.input[0] == name]
    if q:
        s, zp = q[0].input[1], q[0].input[2]
        inits[s].CopyFrom(numpy_helper.from_array(np.array(INPUT_SCALE, np.float32), s))
        inits[zp].CopyFrom(numpy_helper.from_array(np.array(INPUT_ZP, np.int8), zp))
    else:
        g.initializer.extend([numpy_helper.from_array(np.array(INPUT_SCALE, np.float32), "board_input_scale"),
                              numpy_helper.from_array(np.array(INPUT_ZP, np.int8), "board_input_zp")])
        for n in g.node:
            for i, x in enumerate(n.input):
                if x == name:
                    n.input[i] = name + "_board_dq"
        g.node.insert(0, helper.make_node("DequantizeLinear", [name + "_board_q", "board_input_scale",
                                                              "board_input_zp"], [name + "_board_dq"],
                                          name="board_input_dq"))
        g.node.insert(0, helper.make_node("QuantizeLinear", [name, "board_input_scale", "board_input_zp"],
                                          [name + "_board_q"], name="board_input_q"))

    out = g.output[0].name
    dq = [n for n in g.node if n.op_type == "DequantizeLinear" and n.output[0] == out]
    inits = {t.name: t for t in g.initializer}
    if dq:
        # the board reads the logits with zero point 0 (network_shape.h): make the pair symmetric
        scale = float(numpy_helper.to_array(inits[dq[0].input[1]]))
        zp = int(numpy_helper.to_array(inits[dq[0].input[2]])) if len(dq[0].input) > 2 else 0
        out_scale = max(abs(-128 - zp), abs(127 - zp)) * scale / 127.0
        for n in g.node:
            if n.op_type in ("QuantizeLinear", "DequantizeLinear") and n.input[1] == dq[0].input[1]:
                inits[n.input[1]].CopyFrom(numpy_helper.from_array(np.array(out_scale, np.float32), n.input[1]))
                if len(n.input) > 2:
                    inits[n.input[2]].CopyFrom(numpy_helper.from_array(np.array(0, np.int8), n.input[2]))
    else:
        onnx.save(m, qdq_path)
        sess = ort.InferenceSession(qdq_path, providers=["CPUExecutionProvider"])
        _, in_dims, _ = graph_layout(qdq_path)
        peak = max(float(np.abs(sess.run(None, {name: window_to_input(w, in_dims)})[0]).max()) for w in windows)
        out_scale = max(peak, 1e-6) / 127.0
        producer = [n for n in g.node if out in n.output][0]
        producer.output[list(producer.output).index(out)] = out + "_float"
        g.initializer.extend([numpy_helper.from_array(np.array(out_scale, np.float32), "board_output_scale"),
                              numpy_helper.from_array(np.array(0, np.int8), "board_output_zp")])
        g.node.extend([
            helper.make_node("QuantizeLinear", [out + "_float", "board_output_scale", "board_output_zp"],
                             [out + "_q"], name="board_output_q"),
            helper.make_node("DequantizeLinear", [out + "_q", "board_output_scale", "board_output_zp"], [out],
                             name="board_output_dq")])
    onnx.checker.check_model(m)
    onnx.save(m, qdq_path)
    return INPUT_SCALE, out_scale


class BoardSim:
    """An ONNX model run the way the board runs it: norm.c windows, committed steps, greedy CTC."""

    def __init__(self, path):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = os.cpu_count() or 2
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.name, self.in_dims, self.out_dims = graph_layout(path)
        self.left = LEFT_FRAMES // (WIN_FRAMES // WIN_STEPS)
        self.core = STRIDE_FRAMES // (WIN_FRAMES // WIN_STEPS)

    def run_windows(self, windows):
        """int8 windows [600, 64], one per 50 frames as norm.c writes them -> text, as net_ctc.c
        commits it: the first window from its start, every later one only its core."""
        ids = []
        for k, w in enumerate(windows):
            y = self.sess.run(None, {self.name: window_to_input(w, self.in_dims)})[0]
            steps = output_to_steps(y, self.out_dims).argmax(-1)
            first = 0 if k == 0 else self.left
            ids += steps[first: self.left + self.core].tolist()
        return greedy_text(ids)

    def transcribe(self, pcm, lead_s=1.0, tail_s=2.5, noise_std=30.0):
        """One stream: room noise (about -60 dBFS), the audio, room noise; the tail lets the
        decoder commit the last words (they are committed 1.3 s after they are heard)."""
        rng = np.random.default_rng(0)
        noise = lambda n: np.round(rng.standard_normal(n) * noise_std).astype(np.int16)
        x = np.concatenate([noise(int(lead_s * SR)), pcm, noise(int(tail_s * SR))])
        return self.run_windows(w for _, w in board_windows(logmel_numpy(x)))

    def evaluate(self, items, gap_s=0.6, noise_std=30.0):
        """The utterances back to back as one conversation, as the board hears speech: its
        statistics then come from the last 30 s of speech and room, not from a cold start."""
        rng = np.random.default_rng(1)
        parts = []
        for pcm, _ in items:
            parts += [pcm, np.round(rng.standard_normal(int(gap_s * SR)) * noise_std).astype(np.int16)]
        hyp = self.transcribe(np.concatenate(parts), noise_std=noise_std)
        return wer([" ".join(t for _, t in items)], [hyp])


NETWORK_SHAPE_H = """/**
 * @file     network_shape.h
 * @brief    Interface of the compiled CynipsNet network ({name}).
 * @details  Fine-tuned for Indian-accented English by model-tools/asr_finetune (Colab), from
 *           cynipsnet_punct1: same layers, so the same memory pools and weight size.
 *           asr_model_params.h takes the model geometry from this interface.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef ASR_ENGINE_NETWORK_SHAPE_H
#define ASR_ENGINE_NETWORK_SHAPE_H

#ifdef __cplusplus
extern "C" {{
#endif

/** @brief Public name of the compiled network. */
#define ASR_NETWORK_NAME "{name}"

/** @brief Input frames: the tensor is int8 [frames][bands]. */
#define ASR_NETWORK_INPUT_FRAMES 600U

/** @brief Bands per input frame. */
#define ASR_NETWORK_INPUT_BANDS 64U

/** @brief Quantization scale of the input. */
#define ASR_NETWORK_INPUT_SCALE {in_scale:.10f}f

/** @brief Quantization zero point of the input. */
#define ASR_NETWORK_INPUT_ZERO_POINT 0

/** @brief Output time steps: the tensor is int8 [steps][classes]. */
#define ASR_NETWORK_OUTPUT_STEPS 300U

/** @brief Output classes per step. */
#define ASR_NETWORK_OUTPUT_CLASSES 32U

/** @brief Character of every output class: CynipsNet's 29, then the marks (asr_model_params.h). */
#define ASR_MODEL_NET_TOKENS "_ ABCDEFGHIJKLMNOPQRSTUVWXYZ',.?"

/** @brief First class that is a punctuation mark; the display shows sentence case. */
#define ASR_MODEL_NET_PUNCT_FIRST_INDEX 29U

/** @brief Quantization scale of the output; the zero point is 0. */
#define ASR_NETWORK_OUTPUT_SCALE {out_scale:.9f}f

/**
 * @brief Bytes from the input tensor to the end of its memory pool (npuRAM5).
 * @note The input tensor starts at 0x342E0000: check that the new report says the same.
 */
#define ASR_NETWORK_INPUT_POOL_BYTES 458752U

#ifdef __cplusplus
}}
#endif

#endif /* ASR_ENGINE_NETWORK_SHAPE_H */
"""


def write_network_shape(path, name, in_scale, out_scale):
    with open(path, "w") as f:
        f.write(NETWORK_SHAPE_H.format(name=name, in_scale=in_scale, out_scale=out_scale))
