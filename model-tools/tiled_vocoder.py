"""MMS-TTS HiFi-GAN as an NPU-friendly static graph: rows of <= 768 samples, valid convolutions.

The Neural-ART NPU runs a (1 x k) convolution in hardware only if the row is at most ~768 wide;
HiFi-GAN's later stages are 4096-16384 samples long. So at each of the four upsampling stages the
signal is cut into rows of ROW samples, each carrying the halo its convolutions need from its
neighbours; all convolutions are "valid" (no padding) and shrink the rows back to their core.
Between stages the rows are flattened (a free reshape in the NPU's channel-last layout) and
re-gathered with overlap (Slice + Concat).

The graph turns latent frames [-PAD, F + PAD) into audio samples [0, F * 256) of the chunk. Away
from the utterance's ends this equals running the original decoder on the whole utterance (the
original zero-pads every layer at the ends; here the CPU zero-pads the latents instead).

Plan (all coordinates relative to the chunk start, at each stage's own rate):
  stage s: upsample x u (3-tap polyphase conv + interleave), then three residual blocks whose
  widest (k=11, dilations 1,3,5) needs 60 samples each side; stage 4 also feeds conv_post (k=7).
"""
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from npu_vocoder_w import Polyphase, conv_w  # noqa: E402

MAX_ROW = 640  # core width; with the 2 x 64 halo a row stays within the NPU's ~768


def ceil_to(x, m):
    return int(math.ceil(x / m) * m)


class Stage:
    """Row layout of one upsampling stage (all in output-rate samples unless noted)."""

    def __init__(self, u, out_lo, out_hi, extra_post=0):
        self.u = u
        self.halo = ceil_to(60 + extra_post, u)            # samples each side before the res blocks
        self.post = extra_post
        self.out_lo = int(math.floor(out_lo / u)) * u      # aligned to u
        span = out_hi - self.out_lo
        self.rows = int(math.ceil(span / MAX_ROW))
        self.row = ceil_to(span / self.rows, u)            # core width of every row
        self.out_hi = self.out_lo + self.rows * self.row
        # input-rate interval each row needs: core/u + halo/u each side + 1 for the 3-tap conv
        self.in_w = self.row // u + 2 * (self.halo // u) + 2
        self.in_lo = self.out_lo // u - self.halo // u - 1
        self.in_hi = self.in_lo + (self.rows - 1) * (self.row // u) + self.in_w


def plan(frames):
    """Stages from the last to the first; returns (stages first..last, latent frames needed)."""
    s4 = Stage(2, 0, frames * 256, extra_post=3)
    s3 = Stage(2, s4.in_lo, s4.in_hi)
    s2 = Stage(8, s3.in_lo, s3.in_hi)
    s1 = Stage(8, s2.in_lo, s2.in_hi)
    pad = max(-(s1.in_lo - 3), (s1.in_hi + 3) - frames)  # conv_pre (k=7) needs 3 frames each side
    return [s1, s2, s3, s4], pad


class SplitDilated(nn.Module):
    """Valid Conv2d (1 x k), dilation d > 1, k > 3, as a sum of <= 3-tap dilated convs on slices.
    Each group of taps j0..j0+2 reads the input from offset j0 * d; the bias is added once."""

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        k, d = conv.kernel_size[1], conv.dilation[1]
        self.d, self.span = d, (k - 1) * d
        self.parts = nn.ModuleList()
        self.offsets = []
        for j0 in range(0, k, 3):
            taps = min(3, k - j0)
            c = nn.Conv2d(conv.in_channels, conv.out_channels, (1, taps), dilation=(1, d), bias=(j0 == 0))
            with torch.no_grad():
                c.weight.copy_(conv.weight[:, :, :, j0:j0 + taps])
                if j0 == 0:
                    c.bias.copy_(conv.bias)
            self.parts.append(c)
            self.offsets.append(j0 * d)

    def forward(self, x):
        n = x.shape[-1] - self.span
        y = None
        for c, off in zip(self.parts, self.offsets):
            p = c(x[..., off:])[..., :n]
            y = p if y is None else y + p
        return y


def npu_conv(conv: nn.Conv2d):
    k, d = conv.kernel_size[1], conv.dilation[1]
    return SplitDilated(conv) if (d > 1 and k > 3) else conv


class TiledVocoder(nn.Module):
    """latents [1, 192, frames + 2*pad] -> audio [1, 1, 1, frames * 256]."""

    def __init__(self, dec, slope, frames):
        super().__init__()
        self.frames, self.slope, self.k = frames, slope, dec.num_kernels
        self.stages, self.pad = plan(frames)
        self.conv_pre = conv_w(dec.conv_pre)
        self.conv_pre.padding = (0, 0)
        self.ups = nn.ModuleList(Polyphase(u, 1.0 if i == 0 else 1.0 / self.k) for i, u in enumerate(dec.upsampler))
        for up in self.ups:
            up.conv.padding = (0, 0)
        self.res = nn.ModuleList()
        for b in dec.resblocks:
            blk = nn.ModuleList()
            for c1, c2 in zip(b.convs1, b.convs2):
                a, bb = conv_w(c1), conv_w(c2)
                a.padding, bb.padding = (0, 0), (0, 0)
                blk.append(nn.ModuleList([npu_conv(a), bb]))
            self.res.append(blk)
        self.conv_post = conv_w(dec.conv_post)
        self.conv_post.padding = (0, 0)
        with torch.no_grad():
            self.conv_post.weight.mul_(1.0 / self.k)

    @staticmethod
    def crop(x, n):
        return x if n == 0 else x[..., n:x.shape[-1] - n]

    @staticmethod
    def upsample(up, x):
        """3-tap polyphase conv (valid) + interleave: [C, R, n] -> [C', R, (n - 2) * s].
        Channel-last, the interleave is a plain reshape: [R, n', s*C'] -> [R, n'*s, C']."""
        y = up.conv(x)
        b, _, r, n = y.shape
        y = y.view(b, up.s, up.cout, r, n).permute(0, 2, 3, 4, 1)
        return y.reshape(b, up.cout, r, n * up.s)

    def resblock(self, blk, x):
        for a, b in blk:
            h = b(F.leaky_relu(a(F.leaky_relu(x, self.slope)), self.slope))
            x = self.crop(x, (x.shape[-1] - h.shape[-1]) // 2) + h
        return x

    def forward(self, latents):
        # stage 0: conv_pre over the flat latent row; coordinates [-pad + 3, frames + pad - 3)
        x = self.conv_pre(latents.unsqueeze(2))
        lo = -self.pad + 3
        for i, st in enumerate(self.stages):
            # gather overlapping rows from the flat signal x (coordinate lo at index 0)
            rows = [x[..., st.in_lo + r * (st.row // st.u) - lo: st.in_lo + r * (st.row // st.u) - lo + st.in_w]
                    for r in range(st.rows)]
            t = torch.cat(rows, dim=2) if len(rows) > 1 else rows[0]          # [C, rows, in_w]
            t = self.upsample(self.ups[i], F.leaky_relu(t, self.slope))      # [C', rows, (in_w-2)*u]
            outs = [self.resblock(self.res[i * self.k + j], t) for j in range(self.k)]
            w = min(o.shape[-1] for o in outs)
            s = None
            for o in outs:
                o = self.crop(o, (o.shape[-1] - w) // 2)
                s = o if s is None else s + o
            # now [C', rows, ROW + 2*post]: keep the core (+ conv_post halo on the last stage)
            s = self.crop(s, (s.shape[-1] - (st.row + 2 * st.post)) // 2)
            if i < len(self.stages) - 1:
                x = s.reshape(s.shape[0], s.shape[1], 1, -1)                  # flatten rows (free)
                lo = st.out_lo
            else:
                y = torch.tanh(self.conv_post(F.leaky_relu(s, 0.01)))         # [1, 1, rows, ROW]
                y = y.reshape(1, 1, 1, -1)
                start = -st.out_lo
                return y[..., start:start + self.frames * 256]


if __name__ == "__main__":


    from transformers import VitsModel

    m = VitsModel.from_pretrained(sys.argv[1]).eval()
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    tv = TiledVocoder(m.decoder, m.config.leaky_relu_slope, frames).eval()
    print("frames %d, latent pad %d, rows per stage %s, row cores %s, row input widths %s" %
          (frames, tv.pad, [s.rows for s in tv.stages], [s.row for s in tv.stages], [s.in_w for s in tv.stages]))
    total = 4 * frames + 2 * tv.pad
    z = torch.randn(1, 192, total)
    with torch.no_grad():
        full = m.decoder(z).flatten()
        start = tv.pad + frames  # check a chunk in the middle of a longer utterance
        chunk = tv(z[:, :, start - tv.pad:start + frames + tv.pad]).flatten()
        ref = full[start * 256:(start + frames) * 256]
    print("chunk vs full-utterance decoder: max |diff| = %.2e (peak %.3f)" %
          (float((chunk - ref).abs().max()), float(ref.abs().max())))
