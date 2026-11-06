"""PyTorch layers for the Neural Engine's preferred layout, channels-first (B, C, 1, S). All keep the original weights."""
import torch
import torch.nn as nn


def conv1x1(weight, bias=None):
    """Conv2d equal to F.linear(x, weight, bias), for channels-first input."""
    c = nn.Conv2d(weight.shape[1], weight.shape[0], 1, bias=bias is not None)
    c.weight.data = weight[:, :, None, None].clone()
    if bias is not None:
        c.bias.data = bias.clone()
    return c


def factor_patch_conv(conv, first):
    """Replace a patch-embedding conv (kernel = stride = k) with an exact two-stage equivalent.

    Stage one uses identity kernels of size `first` to move pixels into channels. Stage two holds the original weights,
    with kernel k / first. On PE-Core-L14 this took an M2 Pro's Neural Engine from 83.6 to 73.5 ms, and it costs almost nothing.
    """
    k, c, out = conv.kernel_size[0], conv.in_channels, conv.out_channels
    if conv.kernel_size[0] != conv.kernel_size[1] or tuple(conv.stride) != tuple(conv.kernel_size) or conv.groups != 1:
        raise ValueError("factor_patch_conv needs a square, ungrouped conv with stride equal to its kernel")
    if k % first or first in (1, k):
        raise ValueError(f"first={first} must be a proper factor of the kernel size {k}")
    second, middle = k // first, c * first * first
    unshuffle = nn.Conv2d(c, middle, first, first, bias=False)
    project = nn.Conv2d(middle, out, second, second, bias=conv.bias is not None)
    unshuffle.weight.data.zero_()
    for i in range(c):
        for y in range(first):
            for x in range(first):
                ch = (i * first + y) * first + x
                unshuffle.weight.data[ch, i, y, x] = 1
                for Y in range(second):
                    for X in range(second):
                        project.weight.data[:, ch, Y, X] = conv.weight.data[:, i, first * Y + y, first * X + x]
    if conv.bias is not None:
        project.bias.data = conv.bias.data.clone()
    return nn.Sequential(unshuffle, project)


class LayerNormCF(nn.Module):
    """nn.LayerNorm over channel axis 1. CoreML's layer_norm needs trailing axes, so this is spelled out."""

    def __init__(self, ln):
        super().__init__()
        self.eps = ln.eps
        self.gamma = nn.Parameter(ln.weight.data[None, :, None, None].clone())
        self.beta = nn.Parameter(ln.bias.data[None, :, None, None].clone())

    def forward(self, x):
        z = x - x.mean(1, keepdim=True)
        var = (z * z).mean(1, keepdim=True)
        return z / torch.sqrt(var + self.eps) * self.gamma + self.beta


class SplitOut(nn.Module):
    """F.linear with a wide output, as one conv per chunk of `width` output channels. Returns the chunks.

    Use this when the full output would pass the on-chip working-set limit, such as an MLP's expansion layer.
    """

    def __init__(self, weight, bias, width):
        super().__init__()
        bias = bias.split(width) if bias is not None else [None] * (weight.shape[0] // width)
        self.convs = nn.ModuleList(conv1x1(w, b) for w, b in zip(weight.split(width), bias))

    def forward(self, x):
        return [c(x) for c in self.convs]


class SplitIn(nn.Module):
    """F.linear with a wide input given as chunks of `width` channels, as one conv per chunk with the outputs summed."""

    def __init__(self, weight, bias, width):
        super().__init__()
        self.convs = nn.ModuleList(conv1x1(w) for w in weight.split(width, dim=1))
        self.bias = None if bias is None else nn.Parameter(bias[None, :, None, None].clone())

    def forward(self, xs):
        y = 0 if self.bias is None else self.bias
        for c, x in zip(self.convs, xs):
            y = y + c(x)
        return y


class SplitMLP(nn.Module):
    """proj(activation(fc(x))) with the hidden layer kept as branches of `width` channels. Each element is computed identically."""

    def __init__(self, fc_weight, fc_bias, proj_weight, proj_bias, width=None, activation=nn.functional.gelu):
        super().__init__()
        width = width or proj_weight.shape[0]
        self.fc = SplitOut(fc_weight, fc_bias, width)
        self.proj = SplitIn(proj_weight, proj_bias, width)
        self.activation = activation

    def forward(self, x):
        return self.proj([self.activation(t) for t in self.fc(x)])


class RotateHalf(nn.Module):
    """RoPE's rotate_half on channels, as a constant grouped conv: each pair (a, b) maps to (-b, a)."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1, groups=channels // 2, bias=False)
        w = torch.tensor([[0.0, -1.0], [1.0, 0.0]]).repeat(channels // 2, 1)
        self.conv.weight.data = w[:, :, None, None]

    def forward(self, x):
        return self.conv(x)


def attend(q, k, v, scale):
    """Attention for q, k, v of shape (B, h, d, S). k^T q is (B, h, Sk, Sq), so softmax over Sk and v @ W needs no more transposes."""
    w = torch.matmul(k.transpose(-1, -2), q) * scale
    return torch.matmul(v, w.softmax(dim=-2))


def attend_per_head(q, k, v, scale):
    """attend() run once per head (Apple's "split einsum"). Smaller tensors stay on chip, and softmax runs per head."""
    parts = [attend(qi, ki, vi, scale) for qi, ki, vi in zip(q.split(1, dim=1), k.split(1, dim=1), v.split(1, dim=1))]
    return torch.cat(parts, dim=1)
