"""MMS-TTS HiFi-GAN laid out for the Neural-ART NPU: time on the width axis, no ConvTranspose.

Numerically equal to the original decoder (to float rounding). Rewrites:
- Every Conv1d becomes a Conv2d over [B, C, 1, L] with a (1 x k) kernel: the NPU limits vertical
  kernels to 3 but handles wide rows and wide kernels.
- Each ConvTranspose1d (stride s, kernel 2s, padding s/2) becomes a 3-tap polyphase Conv2d with
  s * C_out output channels (channel = phase * C_out + c), then an interleave to length L * s. In
  the NPU's channel-last layout that interleave is a plain reshape.
- The 1/3 averaging of the residual blocks is folded into the next layer's weights (exact:
  leaky_relu is positively homogeneous and the next layer is linear).
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_w(conv: nn.Conv1d) -> nn.Conv2d:
    k, d = conv.kernel_size[0], conv.dilation[0]
    c = nn.Conv2d(conv.in_channels, conv.out_channels, (1, k), padding=(0, conv.padding[0]), dilation=(1, d),
                  bias=conv.bias is not None)
    with torch.no_grad():
        c.weight.copy_(conv.weight.unsqueeze(2))
        if conv.bias is not None:
            c.bias.copy_(conv.bias)
    return c


class Polyphase(nn.Module):
    """ConvTranspose1d(C_in, C_out, K=2s, stride=s, padding=s/2) as Conv2d(1x3) + interleave."""

    def __init__(self, ct: nn.ConvTranspose1d, scale: float = 1.0):
        super().__init__()
        s, K, p = ct.stride[0], ct.kernel_size[0], ct.padding[0]
        assert K == 2 * s and p == s // 2, (K, s, p)
        cin, cout = ct.in_channels, ct.out_channels
        self.s, self.cout = s, cout
        self.conv = nn.Conv2d(cin, s * cout, (1, 3), padding=(0, 1))
        w = torch.zeros(s * cout, cin, 1, 3)
        wt = ct.weight.detach() * scale  # [cin, cout, K]
        for r in range(s):
            for o in (-1, 0, 1):
                k = -o * s + r + p
                if 0 <= k < K:
                    w[r * cout:(r + 1) * cout, :, 0, o + 1] = wt[:, :, k].t()
        with torch.no_grad():
            self.conv.weight.copy_(w)
            self.conv.bias.copy_(ct.bias.detach().repeat(s))

    def forward(self, x):
        b, _, _, n = x.shape
        y = self.conv(x)                                   # [B, s*C, 1, n], channel = r*C + c
        y = y.view(b, self.s, self.cout, 1, n).permute(0, 2, 3, 4, 1)
        return y.reshape(b, self.cout, 1, n * self.s)      # out[c, m*s + r] = y[r*C + c, m]


class ResBlockW(nn.Module):
    def __init__(self, block, slope):
        super().__init__()
        self.c1 = nn.ModuleList(conv_w(c) for c in block.convs1)
        self.c2 = nn.ModuleList(conv_w(c) for c in block.convs2)
        self.slope = slope

    def forward(self, x):
        for a, b in zip(self.c1, self.c2):
            x = x + b(F.leaky_relu(a(F.leaky_relu(x, self.slope)), self.slope))
        return x


class NpuVocoderW(nn.Module):
    """Input latents [B, 192, L]; output audio [B, 1, 1, 256 * L]."""

    def __init__(self, dec, slope):
        super().__init__()
        dec = copy.deepcopy(dec)
        self.slope, self.k = slope, dec.num_kernels
        self.conv_pre = conv_w(dec.conv_pre)
        self.ups = nn.ModuleList(Polyphase(u, 1.0 if i == 0 else 1.0 / self.k) for i, u in enumerate(dec.upsampler))
        self.res = nn.ModuleList(ResBlockW(b, slope) for b in dec.resblocks)
        self.conv_post = conv_w(dec.conv_post)
        with torch.no_grad():
            self.conv_post.weight.mul_(1.0 / self.k)

    def forward(self, latents):
        x = self.conv_pre(latents.unsqueeze(2))
        for i, up in enumerate(self.ups):
            x = up(F.leaky_relu(x, self.slope))
            s = self.res[i * self.k](x)
            for j in range(1, self.k):
                s = s + self.res[i * self.k + j](x)
            x = s
        return torch.tanh(self.conv_post(F.leaky_relu(x, 0.01)))


if __name__ == "__main__":
    import sys

    from transformers import VitsModel

    m = VitsModel.from_pretrained(sys.argv[1]).eval()
    v = NpuVocoderW(m.decoder, m.config.leaky_relu_slope).eval()
    for frames in (60, 64):
        x = torch.randn(1, 192, frames)
        with torch.no_grad():
            a, b = m.decoder(x).flatten(), v(x).flatten()
        print("frames %d: max |orig - npu| = %.2e (peak %.3f)" % (frames, float((a - b).abs().max()), float(a.abs().max())))
