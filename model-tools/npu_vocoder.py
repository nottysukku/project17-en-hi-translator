"""MMS-TTS HiFi-GAN rewritten for the STM32N6 Neural-ART NPU, numerically equal to the original.

Two exact rewrites remove the operations the NPU compiler ran on the CPU:
- Dilated convolutions (dilation 3 and 5 in the residual blocks): a dilation-d convolution over
  length L equals a plain (k x 1) 2-D convolution over the input viewed as (L/d, d). In the
  NPU's channel-last layout that view is free. Needs L divisible by d at every stage, so the
  chunk length in latent frames must be a multiple of 15 (the stages are 8x, 64x, 128x, 256x).
- The "/ 3" averaging of the three residual blocks: leaky_relu is positively homogeneous and the
  next layer is linear, so 1/3 is folded into the weights of the next upsampler (or conv_post).
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedAsReshape(nn.Module):
    """Conv1d(C, C, k, dilation=d, padding=d*(k-1)//2) as reshape -> Conv2d((k, 1)) -> reshape."""

    def __init__(self, conv: nn.Conv1d):
        super().__init__()
        self.d = conv.dilation[0]
        k = conv.kernel_size[0]
        self.conv = nn.Conv2d(conv.in_channels, conv.out_channels, (k, 1), padding=((k - 1) // 2, 0))
        with torch.no_grad():
            self.conv.weight.copy_(conv.weight.unsqueeze(-1))
            self.conv.bias.copy_(conv.bias)

    def forward(self, x):
        b, c, n = x.shape
        y = self.conv(x.view(b, c, n // self.d, self.d))
        return y.reshape(b, -1, n)


def plain(conv: nn.Conv1d):
    return conv if conv.dilation[0] == 1 else DilatedAsReshape(conv)


class ResBlock(nn.Module):
    def __init__(self, block, slope):
        super().__init__()
        self.convs1 = nn.ModuleList(plain(c) for c in block.convs1)
        self.convs2 = nn.ModuleList(plain(c) for c in block.convs2)
        self.slope = slope

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            h = c2(F.leaky_relu(c1(F.leaky_relu(x, self.slope)), self.slope))
            x = x + h
        return x


class NpuVocoder(nn.Module):
    def __init__(self, dec, slope):
        super().__init__()
        dec = copy.deepcopy(dec)
        self.slope = slope
        self.k = dec.num_kernels
        self.conv_pre = dec.conv_pre
        self.ups = dec.upsampler
        self.res = nn.ModuleList(ResBlock(b, slope) for b in dec.resblocks)
        self.conv_post = dec.conv_post
        with torch.no_grad():  # fold the 1/num_kernels of every stage into the next linear layer
            for up in list(self.ups)[1:]:
                up.weight.mul_(1.0 / self.k)
            self.conv_post.weight.mul_(1.0 / self.k)

    def forward(self, latents):
        x = self.conv_pre(latents)
        for i, up in enumerate(self.ups):
            x = up(F.leaky_relu(x, self.slope))
            s = self.res[i * self.k](x)
            for j in range(1, self.k):
                s = s + self.res[i * self.k + j](x)
            x = s
        # The original uses leaky_relu's default slope (0.01) here.
        return torch.tanh(self.conv_post(F.leaky_relu(x, 0.01)))


if __name__ == "__main__":
    import sys

    from transformers import VitsModel

    m = VitsModel.from_pretrained(sys.argv[1]).eval()
    v = NpuVocoder(m.decoder, m.config.leaky_relu_slope).eval()
    for frames in (60, 120):
        x = torch.randn(1, 192, frames)
        with torch.no_grad():
            a, b = m.decoder(x), v(x)
        print("frames %d: max |orig - npu| = %.2e (output peak %.3f)" % (frames, float((a - b).abs().max()),
                                                                        float(a.abs().max())))
