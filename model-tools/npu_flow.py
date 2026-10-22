"""MMS-TTS prior flow (reverse direction) as an NPU-friendly static graph.

Input: prior latents [1, 192, L] and the frame mask [1, 1, L] (1 inside the utterance);
output: the latents the vocoder decodes. L is fixed (512 frames = 8.2 s, so the mask's
width-broadcast stays within the NPU's 512-element limit).

Rewrites, all exact:
- time on the width axis, (1 x k) kernels (the NPU limits vertical kernels to 3);
- the channel flip before every coupling layer is folded into the weights: the flow keeps the
  channels in their stored order and alternates which half conditions and which is updated,
  reversing the channel order of the affected weights instead.
"""
import torch
import torch.nn as nn

H = 192          # flow / hidden width
HALF = H // 2


def w2(conv: nn.Conv1d, in_perm=None, out_perm=None):
    """Conv1d -> Conv2d (1 x k) with optional input/output channel permutations."""
    w = conv.weight.detach()
    b = conv.bias.detach() if conv.bias is not None else None
    if in_perm is not None:
        w = w[:, in_perm, :]
    if out_perm is not None:
        w = w[out_perm]
        b = b[out_perm] if b is not None else None
    c = nn.Conv2d(w.shape[1], w.shape[0], (1, w.shape[2]), padding=(0, conv.padding[0]),
                  dilation=(1, conv.dilation[0]), bias=b is not None)
    with torch.no_grad():
        c.weight.copy_(w.unsqueeze(2))
        if b is not None:
            c.bias.copy_(b)
    return c


class Coupling(nn.Module):
    def __init__(self, layer, flipped):
        super().__init__()
        rev = torch.arange(HALF - 1, -1, -1)
        self.flipped = flipped
        # flipped: the conditioning half is stored channels 96..191 reversed; updated: 0..95 reversed
        self.pre = w2(layer.conv_pre, in_perm=rev if flipped else None)
        wn = layer.wavenet
        self.ins = nn.ModuleList(w2(c) for c in wn.in_layers)
        self.res = nn.ModuleList(w2(c) for c in wn.res_skip_layers)
        self.post = w2(layer.conv_post, out_perm=rev if flipped else None)

    def forward(self, x, mask):
        cond, upd = (x[:, HALF:], x[:, :HALF]) if self.flipped else (x[:, :HALF], x[:, HALF:])
        h = self.pre(cond) * mask
        out = None
        n = len(self.ins)
        for i in range(n):
            a = self.ins[i](h)
            acts = torch.tanh(a[:, :H]) * torch.sigmoid(a[:, H:])
            rs = self.res[i](acts)
            if i < n - 1:
                h = (h + rs[:, :H]) * mask
                skip = rs[:, H:]
            else:
                skip = rs
            out = skip if out is None else out + skip
        mean = self.post(out * mask) * mask
        upd = (upd - mean) * mask
        return torch.cat([upd, cond], dim=1) if self.flipped else torch.cat([cond, upd], dim=1)


class NpuFlow(nn.Module):
    """latents [1, 192, L], mask [1, 1, L] -> [1, 192, 1, L]."""

    def __init__(self, flow):
        super().__init__()
        layers = list(flow.flows)[::-1]          # reverse direction: last coupling first
        self.layers = nn.ModuleList(Coupling(l, flipped=(i % 2 == 0)) for i, l in enumerate(layers))

    def forward(self, latents, mask):
        x, m = latents.unsqueeze(2), mask.unsqueeze(2)
        for c in self.layers:
            x = c(x, m)
        return x


if __name__ == "__main__":
    import sys

    from transformers import VitsModel

    mdl = VitsModel.from_pretrained(sys.argv[1]).eval()
    f = NpuFlow(mdl.flow).eval()
    L, valid = 512, 300
    z = torch.randn(1, H, L)
    mask = torch.zeros(1, 1, L)
    mask[..., :valid] = 1
    z = z * mask
    with torch.no_grad():
        ref = mdl.flow(z, mask, None, reverse=True)
        out = f(z, mask).squeeze(2)
    print("flow: max |orig - npu| = %.2e (peak %.3f)" % (float((ref - out).abs().max()), float(ref.abs().max())))
