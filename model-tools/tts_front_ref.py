"""Reference of the MMS-TTS Hindi front end as the STM32N6 runs it on the CPU (tts_front.c).

Text -> character ids (VitsTokenizer, add_blank) -> text encoder (6 post-norm transformer layers
with windowed relative attention) -> prior means / log-variances, and the stochastic duration
predictor run in reverse (rational-quadratic spline flows) -> durations ceil(exp(logw)) -> the
prior latents expanded to frames, plus noise: z = m + eps * exp(logs) * 0.667. The flow and the
HiFi-GAN vocoder (on the NPU) turn z into audio.

Written as plain loops over the same steps as the C code, so each intermediate can be compared.
`check()` runs it against transformers' VitsModel with the same noise.

    python tts_front_ref.py <mms_dir>            # float reference vs VitsModel
"""
import json
import math
import os
import sys
import unicodedata

import numpy as np
import torch
import torch.nn.functional as F

NOISE_SCALE, NOISE_SCALE_DUR, TAIL, BINS = 0.667, 0.8, 5.0, 10
GROUP = 64

# Precomposed Devanagari letters with a nukta, and letters MMS-TTS has no symbol for. The model's
# alphabet has the nukta only as a combining mark (U+093C), so U+0958..U+095F would be dropped.
DECOMPOSE = {chr(c): unicodedata.normalize("NFD", chr(c)) for c in list(range(0x958, 0x960)) + [0x929, 0x931, 0x934]}
REPLACE = {"ऋ": "रि", "ॠ": "री", "ॄ": "ृ"}


def text_to_ids(text, vocab):
    """VitsTokenizer (normalize, add_blank) on Devanagari, after the nukta decomposition above.
    Characters outside the model's alphabet (punctuation, Latin, digits it lacks) are dropped."""
    s = "".join(REPLACE.get(c, DECOMPOSE.get(c, c)) for c in text)
    s = "".join(c if c in vocab else c.lower() for c in s)
    s = "".join(c for c in s if c in vocab).strip()
    ids = [0] * (2 * len(s) + 1)
    ids[1::2] = [vocab[c] for c in s]
    return ids


def fake_quant(w, bits):
    """Symmetric per-64-group quantization along the input dimension (it2_pack.py's Q4 for 4)."""
    q = 2 ** (bits - 1) - 1
    rows, cols = w.shape
    g = w.reshape(rows, cols // GROUP, GROUP)
    s = (g.abs().amax(dim=2, keepdim=True) / q).half().float().clamp_min(1e-8)
    return (torch.round(g / s).clamp(-q, q) * s).reshape(rows, cols)


def act_quant(x):
    """Rows of x [T, K] to int8 with one scale per group of 64, returned dequantized; rounded like
    the C code (multiply by 1/scale, round half away from zero)."""
    t, k = x.shape
    g = x.reshape(t, k // GROUP, GROUP)
    s = g.abs().amax(dim=2, keepdim=True) / 127.0
    s = torch.where(s > 0, s, torch.ones_like(s))
    v = g * (1.0 / s)
    q = torch.where(v >= 0, torch.floor(v + 0.5), torch.ceil(v - 0.5))
    return (q * s).reshape(t, k)


class Front:
    """weights: a VitsModel state dict. wbits: None (float), 8 or 4 for the text encoder's matrices;
    their inputs are then rounded to int8 too, unless quantize_activations is False."""

    def __init__(self, sd, wbits=None, quantize_activations=True):
        self.sd = {k: v.float() for k, v in sd.items()}
        self.wbits = wbits
        self.qact = bool(wbits) and quantize_activations
        self.mat = {}
        for k, v in self.sd.items():
            if not k.startswith("text_encoder.") or not k.endswith(".weight") or v.dim() < 2:
                continue
            if "embed_tokens" in k:
                continue
            # conv [out, in, k] -> [out, k * in], tap-major, the layout the C code multiplies with
            m = v.permute(0, 2, 1).reshape(v.shape[0], -1) if v.dim() == 3 else v
            self.mat[k] = fake_quant(m, wbits) if wbits else m

    def lin(self, x, name):
        """x [T, K] @ W^T + b with the C numerics (int8 activations per group when quantized)."""
        w = self.mat[name + ".weight"]
        if self.qact:
            x = act_quant(x)
        return x @ w.t() + self.sd[name + ".bias"]

    def conv(self, x, name, k):
        """'same' Conv1d over time, x [T, C]: im2col of k taps then one matrix product."""
        t = x.shape[0]
        pad_l, pad_r = (k - 1) // 2, k // 2
        xp = F.pad(x, (0, 0, pad_l, pad_r))
        cols = torch.cat([xp[j:j + t] for j in range(k)], dim=1)
        return self.lin(cols, name)

    # ---- text encoder --------------------------------------------------------------------------
    def attention(self, x, p):
        sd, t = self.sd, x.shape[0]
        heads, hd, win = 2, 96, 4
        q = self.lin(x, p + "q_proj") / math.sqrt(hd)
        k = self.lin(x, p + "k_proj")
        v = self.lin(x, p + "v_proj")
        rk, rv = sd[p + "emb_rel_k"][0], sd[p + "emb_rel_v"][0]  # [2 * win + 1, hd]
        out = torch.zeros(t, heads * hd)
        for h in range(heads):
            qh, kh, vh = q[:, h * hd:(h + 1) * hd], k[:, h * hd:(h + 1) * hd], v[:, h * hd:(h + 1) * hd]
            s = qh @ kh.t()
            for i in range(t):
                for j in range(max(0, i - win), min(t, i + win + 1)):
                    s[i, j] += qh[i] @ rk[j - i + win]
            a = torch.softmax(s, dim=-1)
            o = a @ vh
            for i in range(t):
                for j in range(max(0, i - win), min(t, i + win + 1)):
                    o[i] += a[i, j] * rv[j - i + win]
            out[:, h * hd:(h + 1) * hd] = o
        return self.lin(out, p + "out_proj")

    def text_encoder(self, ids):
        sd = self.sd
        x = sd["text_encoder.embed_tokens.weight"][ids] * math.sqrt(192)
        for i in range(6):
            p = "text_encoder.encoder.layers.%d." % i
            x = F.layer_norm(x + self.attention(x, p + "attention."), (192,),
                             sd[p + "layer_norm.weight"], sd[p + "layer_norm.bias"], 1e-5)
            h = torch.relu(self.conv(x, p + "feed_forward.conv_1", 3))
            h = self.conv(h, p + "feed_forward.conv_2", 3)
            x = F.layer_norm(x + h, (192,), sd[p + "final_layer_norm.weight"], sd[p + "final_layer_norm.bias"], 1e-5)
        stats = self.lin(x, "text_encoder.project")
        return x, stats[:, :192], stats[:, 192:]

    # ---- stochastic duration predictor (reverse) -----------------------------------------------
    def pw(self, x, name):
        """1x1 Conv1d over x [T, C] (float weights: the duration predictor is not quantized)."""
        w = self.sd[name + ".weight"][:, :, 0]
        return x @ w.t() + self.sd[name + ".bias"]

    def dds(self, x, p, g=None):
        sd = self.sd
        if g is not None:
            x = x + g
        t = x.shape[0]
        for i in range(3):
            d = 3 ** i
            w, b = sd[p + "convs_dilated.%d.weight" % i][:, 0, :], sd[p + "convs_dilated.%d.bias" % i]  # [C, 3]
            xp = F.pad(x, (0, 0, d, d))
            h = xp[0:t] * w[:, 0] + xp[d:d + t] * w[:, 1] + xp[2 * d:2 * d + t] * w[:, 2] + b
            h = F.gelu(F.layer_norm(h, (192,), sd[p + "norms_1.%d.weight" % i], sd[p + "norms_1.%d.bias" % i], 1e-5))
            h = self.pw(h, p + "convs_pointwise.%d" % i)
            h = F.gelu(F.layer_norm(h, (192,), sd[p + "norms_2.%d.weight" % i], sd[p + "norms_2.%d.bias" % i], 1e-5))
            x = x + h
        return x

    @staticmethod
    def spline_inverse(y, uw, uh, ud):
        """Inverse of the rational-quadratic spline on [-TAIL, TAIL] (identity outside), one value."""
        if y < -TAIL or y > TAIL:
            return y
        minw = minh = mind = 1e-3
        w = torch.softmax(uw, 0) * (1 - minw * BINS) + minw
        h = torch.softmax(uh, 0) * (1 - minh * BINS) + minh
        cw = torch.cat([torch.zeros(1), torch.cumsum(w, 0)]) * 2 * TAIL - TAIL
        ch = torch.cat([torch.zeros(1), torch.cumsum(h, 0)]) * 2 * TAIL - TAIL
        cw[0], cw[-1], ch[0], ch[-1] = -TAIL, TAIL, -TAIL, TAIL
        w, h = cw[1:] - cw[:-1], ch[1:] - ch[:-1]
        const = math.log(math.exp(1 - mind) - 1)
        d = mind + F.softplus(torch.cat([torch.tensor([const]), ud, torch.tensor([const])]))
        loc = ch.clone()
        loc[-1] += 1e-6
        k = int((y >= loc).sum()) - 1
        delta = h[k] / w[k]
        i1 = d[k] + d[k + 1] - 2 * delta
        i2 = y - ch[k]
        i3 = i2 * i1
        a = h[k] * (delta - d[k]) + i3
        b = h[k] * d[k] - i3
        c = -delta * i2
        disc = b * b - 4 * a * c
        root = (2 * c) / (-b - torch.sqrt(disc))
        return root * w[k] + cw[k]

    def durations(self, hidden, noise):
        """hidden [T, 192] (text encoder output), noise [2, T] ~ N(0,1) -> logw [T]."""
        sd, p = self.sd, "duration_predictor."
        x = self.pw(hidden, p + "conv_pre")
        x = self.dds(x, p + "conv_dds.")
        g = self.pw(x, p + "conv_proj")
        z = noise * NOISE_SCALE_DUR  # [2, T]
        for f in (4, 3, 2):  # reversed conv flows, the first one is never used in reverse
            z = torch.flip(z, [0])
            q = p + "flows.%d." % f
            h = self.pw(z[0][:, None], q + "conv_pre")
            h = self.dds(h, q + "conv_dds.", g)
            h = self.pw(h, q + "conv_proj")  # [T, 29]
            uw, uh, ud = h[:, :BINS] / math.sqrt(192), h[:, BINS:2 * BINS] / math.sqrt(192), h[:, 2 * BINS:]
            z = torch.stack([z[0], torch.stack([self.spline_inverse(z[1][t], uw[t], uh[t], ud[t])
                                                for t in range(z.shape[1])])])
        z = torch.flip(z, [0])
        tr, ls = sd[p + "flows.0.translate"][:, 0], sd[p + "flows.0.log_scale"][:, 0]
        z = (z - tr[:, None]) * torch.exp(-ls)[:, None]
        return z[0]

    @torch.no_grad()
    def run(self, ids, noise_dur, noise_prior_fn):
        """-> durations [T] (frames per input id), z [192, frames]. noise_prior_fn(frames) gives the
        prior noise [192, frames]."""
        hidden, m, logs = self.text_encoder(torch.tensor(ids))
        logw = self.durations(hidden, noise_dur)
        dur = torch.ceil(torch.exp(logw)).long()
        idx = torch.repeat_interleave(torch.arange(len(ids)), dur)
        eps = noise_prior_fn(len(idx))
        z = m[idx].t() + eps * torch.exp(logs[idx].t()) * NOISE_SCALE
        return dur, z, dict(hidden=hidden, m=m, logs=logs, logw=logw)


def check(mms_dir):
    from transformers import AutoTokenizer, VitsModel
    sys.stdout.reconfigure(encoding="utf-8")
    model = VitsModel.from_pretrained(mms_dir).eval()
    tok = AutoTokenizer.from_pretrained(mms_dir)
    vocab = json.load(open(os.path.join(mms_dir, "vocab.json"), encoding="utf-8"))
    front = Front(model.state_dict())
    for text in ["नमस्ते, आप कैसे हैं?", "मुझे समझ नहीं आ रहा है कि आप क्या कह रहे हैं।",
                 "आपकी मदद के लिए बहुत-बहुत धन्यवाद।"]:
        ids = text_to_ids(text, vocab)
        assert ids == tok(text)["input_ids"], "tokenizer differs"
        g = torch.Generator().manual_seed(1)
        nd = torch.randn(2, len(ids), generator=g)
        big = torch.randn(192, 4000, generator=g)
        eps = lambda n: big[:, :n]  # noqa: E731
        dur, z, _ = front.run(ids, nd, eps)
        captured = {}
        hook = model.flow.register_forward_pre_hook(lambda mod, args: captured.__setitem__("z", args[0][0].clone()))
        real_randn, real_like = torch.randn, torch.randn_like
        torch.randn = lambda *s, **k: nd[None].clone()
        torch.randn_like = lambda t, **k: eps(t.shape[-1])[None].clone()
        try:
            with torch.no_grad():
                model(input_ids=torch.tensor([ids]))
        finally:
            torch.randn, torch.randn_like = real_randn, real_like
            hook.remove()
        zr = captured["z"]
        same_len = zr.shape[1] == z.shape[1]
        err = float((zr - z).abs().max()) if same_len else float("nan")
        print("%-50s ids %3d frames %4d (VitsModel %4d)  max |z diff| %.2e" %
              (text, len(ids), z.shape[1], zr.shape[1], err))


if __name__ == "__main__":
    check(sys.argv[1])
