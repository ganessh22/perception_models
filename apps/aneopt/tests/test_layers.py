"""Each rewrite must compute the same function as the layer it replaces."""
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytest

from aneopt.layers import LayerNormCF, RotateHalf, SplitIn, SplitMLP, SplitOut, attend, attend_per_head, conv1x1, factor_patch_conv

torch.manual_seed(0)
C, S = 32, 7
x = torch.randn(2, C, 1, S)
as_linear = lambda t: t.squeeze(2).transpose(1, 2)  # (B, C, 1, S) -> (B, S, C)
as_conv = lambda t: t.transpose(1, 2).unsqueeze(2)


def close(a, b):
    assert torch.allclose(a, b, rtol=1e-4, atol=1e-4), (a - b).abs().max()


@pytest.mark.parametrize("k,first,bias", [(14, 7, False), (14, 2, True), (16, 4, False)])
def test_factor_patch_conv_matches(k, first, bias):
    conv = nn.Conv2d(3, 16, k, k, bias=bias)
    image = torch.randn(1, 3, 4 * k, 2 * k)
    close(factor_patch_conv(conv, first)(image), conv(image))


def test_factor_patch_conv_rejects_other_convs():
    with pytest.raises(ValueError):
        factor_patch_conv(nn.Conv2d(3, 16, 14, 7), 7)
    with pytest.raises(ValueError):
        factor_patch_conv(nn.Conv2d(3, 16, 14, 14), 3)


def test_conv1x1_matches_linear():
    w, b = torch.randn(9, C), torch.randn(9)
    close(conv1x1(w, b)(x), as_conv(F.linear(as_linear(x), w, b)))


def test_layer_norm_matches():
    ln = nn.LayerNorm(C, eps=1e-5)
    nn.init.normal_(ln.weight)
    nn.init.normal_(ln.bias)
    close(LayerNormCF(ln)(x), as_conv(ln(as_linear(x))))


def test_split_out_matches_wide_linear():
    w, b = torch.randn(3 * C, C), torch.randn(3 * C)
    close(torch.cat(SplitOut(w, b, C)(x), dim=1), as_conv(F.linear(as_linear(x), w, b)))


def test_split_in_matches_wide_linear():
    wide = torch.randn(2, 4 * C, 1, S)
    w, b = torch.randn(C, 4 * C), torch.randn(C)
    close(SplitIn(w, b, C)(list(wide.split(C, dim=1))), as_conv(F.linear(as_linear(wide), w, b)))


def test_split_without_bias():
    w = torch.randn(2 * C, C)
    close(torch.cat(SplitOut(w, None, C)(x), dim=1), as_conv(F.linear(as_linear(x), w)))


def test_split_mlp_matches_original():
    fw, fb, pw, pb = torch.randn(4 * C, C), torch.randn(4 * C), torch.randn(C, 4 * C), torch.randn(C)
    reference = F.linear(F.gelu(F.linear(as_linear(x), fw, fb)), pw, pb)
    close(SplitMLP(fw, fb, pw, pb)(x), as_conv(reference))


def test_per_head_attention_matches_all_heads():
    q, k, v = [torch.randn(2, 4, 8, S) for _ in range(3)]
    close(attend_per_head(q, k, v, 0.35), attend(q, k, v, 0.35))


def test_attend_matches_torch():
    q, k, v = [torch.randn(1, 4, 8, S) for _ in range(3)]
    t = lambda a: a.transpose(-1, -2)
    reference = F.scaled_dot_product_attention(t(q), t(k), t(v), scale=8**-0.5)
    close(attend(q, k, v, 8**-0.5), t(reference))


def test_rotate_half():
    a, b = x[:, ::2], x[:, 1::2]
    out = RotateHalf(C)(x)
    close(out[:, ::2], -b)
    close(out[:, 1::2], a)
