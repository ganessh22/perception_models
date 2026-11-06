"""PE-Core-L14-336 vision encoder rewritten for CoreML, using the original weights unchanged.

Modes, so each change's gain can be measured separately:
  A  original (B,S,C) layout and nn.Linear-style ops. Only cleans up attention: q/k/v via split,
     RoPE swap via a constant index_select rather than split/stack per call.
  B  channels-first (B,C,1,S). 1x1 convs instead of linear, RoPE swap as a fixed grouped conv applied
     before the head reshape, attention rearranged to need one transpose rather than several.
  C  B, with attention run per head (Apple's "split einsum" pattern for the ANE).
  D  C, with two changes. Wide QKV, MLP and pooling layers are split into 1024-channel branches, so no activation
     passes the Neural Engine's on-chip limit of about 2 MB. The 14x14 patch conv becomes an exact two-stage conv.
Variants that isolate each part of D: CP is C with the two-stage patch conv. DS is D without it. DM and DQ split only
the MLP or only QKV, also without it.
"""
import math
import sys

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import core.vision_encoder.pe as pe
from aneopt.layers import LayerNormCF, RotateHalf, SplitIn, SplitOut, attend, conv1x1, factor_patch_conv


def rope_tables(model):
    model.rope.update_grid(torch.device("cpu"), 24, 24)
    return model.rope.freq[0].detach().clone()  # (S, d), row 0 is the zeroed cls token


class AttnA(nn.Module):
    def __init__(self, a, freq):
        super().__init__()
        self.h, self.d, self.e = a.num_heads, a.head_dim, a.embed_dim
        self.w, self.b = nn.Parameter(a.in_proj_weight.data.clone()), nn.Parameter(a.in_proj_bias.data.clone())
        self.out = a.out_proj
        self.scale = a.scale
        self.register_buffer("cos", freq.cos()[None, None])
        # rotate_half(x) = x[perm] * sign, folded into the sin table
        perm = torch.arange(self.d).reshape(-1, 2).flip(1).flatten()
        sign = torch.tensor([-1.0, 1.0]).repeat(self.d // 2)
        self.register_buffer("perm", perm)
        self.register_buffer("sin", (freq.sin() * sign)[None, None])

    def forward(self, x):
        b, s, _ = x.shape
        q, k, v = F.linear(x, self.w, self.b).split(self.e, dim=-1)
        q, k, v = [t.reshape(b, s, self.h, self.d).transpose(1, 2) for t in (q, k, v)]
        q = q * self.cos + q.index_select(-1, self.perm) * self.sin
        k = k * self.cos + k.index_select(-1, self.perm) * self.sin
        a = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.out(a.transpose(1, 2).reshape(b, s, self.e))


class BlockA(nn.Module):
    def __init__(self, r, freq):
        super().__init__()
        self.attn, self.ln_1, self.ln_2, self.mlp = AttnA(r.attn, freq), r.ln_1, r.ln_2, r.mlp

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class AttnB(nn.Module):
    split_heads = False

    def __init__(self, a, freq, split_qkv=False):
        super().__init__()
        self.h, self.d, self.e = a.num_heads, a.head_dim, a.embed_dim
        self.split_qkv = split_qkv
        if split_qkv:
            self.proj = SplitOut(a.in_proj_weight.data, a.in_proj_bias.data, self.e)
        else:
            self.proj = conv1x1(a.in_proj_weight.data, a.in_proj_bias.data)
        self.out = conv1x1(a.out_proj.weight.data, a.out_proj.bias.data)
        self.swap = RotateHalf(self.e)
        self.scale = a.scale
        # (S,d) -> (1,E,1,S), the same rotation for every head
        table = lambda t: t.T.repeat(self.h, 1)[None, :, None, :]
        self.register_buffer("cos", table(freq.cos()))
        self.register_buffer("sin", table(freq.sin()))

    def forward(self, x):
        b, _, _, s = x.shape
        q, k, v = self.proj(x) if self.split_qkv else self.proj(x).split(self.e, dim=1)
        q = q * self.cos + self.swap(q) * self.sin
        k = k * self.cos + self.swap(k) * self.sin
        if self.split_heads:
            parts = [
                attend(qi, ki, vi, self.scale)
                for qi, ki, vi in zip(*[t.reshape(b, self.h, self.d, s).split(1, dim=1) for t in (q, k, v)])
            ]
            a = torch.cat(parts, dim=1)
        else:
            q, k, v = [t.reshape(b, self.h, self.d, s) for t in (q, k, v)]
            a = attend(q, k, v, self.scale)
        return self.out(a.reshape(b, self.e, 1, s))


class AttnC(AttnB):
    split_heads = True


class BlockB(nn.Module):
    def __init__(self, r, freq, attn_cls, split_mlp=False, split_qkv=False):
        super().__init__()
        self.attn = attn_cls(r.attn, freq, split_qkv)
        self.ln_1, self.ln_2 = LayerNormCF(r.ln_1), LayerNormCF(r.ln_2)
        self.split_mlp = split_mlp
        fc, proj = r.mlp.c_fc, r.mlp.c_proj
        if split_mlp:
            self.fc = SplitOut(fc.weight.data, fc.bias.data, proj.out_features)
            self.proj = SplitIn(proj.weight.data, proj.bias.data, proj.out_features)
        else:
            self.fc = conv1x1(fc.weight.data, fc.bias.data)
            self.proj = conv1x1(proj.weight.data, proj.bias.data)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        if self.split_mlp:
            return x + self.proj([F.gelu(t) for t in self.fc(self.ln_2(x))])
        return x + self.proj(F.gelu(self.fc(self.ln_2(x))))


class FastPE(nn.Module):
    def __init__(self, m, mode):
        super().__init__()
        self.mode = mode
        freq = rope_tables(m)
        self.conv1 = factor_patch_conv(m.conv1, 7) if mode in ("D", "CP") else m.conv1
        ap = m.attn_pool
        self.pool_h, self.pool_d = ap.num_heads, ap.embed_dim // ap.num_heads
        e = m.width
        wq, wk, wv = ap.attn.in_proj_weight.data.split(e)
        bq, bk, bv = ap.attn.in_proj_bias.data.split(e)
        # The pooling query depends only on a learned probe, so its projection is a constant.
        q = F.linear(ap.probe.data, wq, bq)[0, 0]

        if mode == "A":
            self.cls = nn.Parameter(m.class_embedding.data[None, None].clone())
            self.pos = nn.Parameter(m.positional_embedding.data[None].clone())
            self.ln_pre, self.ln_post = m.ln_pre, m.ln_post
            self.blocks = nn.ModuleList(BlockA(r, freq) for r in m.transformer.resblocks)
            self.pool_ln = ap.layernorm
            self.pool_mlp = ap.mlp
            self.register_buffer("pool_q", q.reshape(1, self.pool_h, 1, self.pool_d))
            self.pool_kv_w = nn.Parameter(torch.cat([wk, wv]))
            self.pool_kv_b = nn.Parameter(torch.cat([bk, bv]))
            self.pool_out = ap.attn.out_proj
            self.final = nn.Parameter(m.proj.data.clone())
        else:
            attn_cls = AttnB if mode == "B" else AttnC
            split_mlp, split_qkv = mode in ("D", "DS", "DM"), mode in ("D", "DS", "DQ")
            self.split_pool = mode == "D"
            self.cls = nn.Parameter(m.class_embedding.data[None, :, None, None].clone())
            self.pos = nn.Parameter(m.positional_embedding.data.T[None, :, None, :].clone())
            self.ln_pre, self.ln_post = LayerNormCF(m.ln_pre), LayerNormCF(m.ln_post)
            self.blocks = nn.ModuleList(BlockB(r, freq, attn_cls, split_mlp, split_qkv) for r in m.transformer.resblocks)
            self.pool_ln = LayerNormCF(ap.layernorm)
            self.pool_fc = conv1x1(ap.mlp.c_fc.weight.data, ap.mlp.c_fc.bias.data)
            self.pool_proj = conv1x1(ap.mlp.c_proj.weight.data, ap.mlp.c_proj.bias.data)
            self.register_buffer("pool_q", q.reshape(1, self.pool_h, self.pool_d, 1))
            pool_kv = SplitOut if self.split_pool else (lambda w, b, _: conv1x1(w, b))
            self.pool_kv = pool_kv(torch.cat([wk, wv]), torch.cat([bk, bv]), e)
            self.pool_out = conv1x1(ap.attn.out_proj.weight.data, ap.attn.out_proj.bias.data)
            self.final = conv1x1(m.proj.data.T)
        self.e = e
        self.pool_scale = self.pool_d ** -0.5

    def forward(self, x):
        return self.forward_a(x) if self.mode == "A" else self.forward_b(x)

    def forward_a(self, x):
        b = x.shape[0]
        x = self.conv1(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(b, -1, -1), x], dim=1) + self.pos
        x = self.ln_pre(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_post(x)
        s = x.shape[1]
        kv = F.linear(x, self.pool_kv_w, self.pool_kv_b)
        k, v = [t.reshape(b, s, self.pool_h, self.pool_d).transpose(1, 2) for t in kv.split(self.e, dim=-1)]
        a = F.scaled_dot_product_attention(self.pool_q.expand(b, -1, -1, -1), k, v, scale=self.pool_scale)
        y = self.pool_out(a.transpose(1, 2).reshape(b, 1, self.e))
        y = y + self.pool_mlp(self.pool_ln(y))
        return y[:, 0] @ self.final

    def forward_b(self, x):
        b = x.shape[0]
        x = self.conv1(x).flatten(2).unsqueeze(2)
        x = torch.cat([self.cls.expand(b, -1, -1, -1), x], dim=3) + self.pos
        x = self.ln_pre(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_post(x)
        s = x.shape[3]
        kv = self.pool_kv(x) if self.split_pool else self.pool_kv(x).split(self.e, dim=1)
        k, v = [t.reshape(b, self.pool_h, self.pool_d, s) for t in kv]
        a = attend(self.pool_q.expand(b, -1, -1, -1), k, v, self.pool_scale)
        y = self.pool_out(a.reshape(b, self.e, 1, 1))
        y = y + self.pool_proj(F.gelu(self.pool_fc(self.pool_ln(y))))
        return self.final(y).reshape(b, -1)


def main():
    from pathlib import Path

    here = Path(__file__).parent
    modes = sys.argv[1:] or ["A", "B", "C", "D"]
    ref_model = pe.VisionTransformer.from_config("PE-Core-L14-336", pretrained=True).eval()
    val = torch.from_numpy(np.load(here / "data" / "val.npy", mmap_mode="r")[::50].copy())
    with torch.no_grad():
        ref = F.normalize(torch.cat([ref_model(val[i : i + 1]) for i in range(len(val))]), dim=-1)
    for mode in modes:
        m = FastPE(ref_model, mode).eval()
        with torch.no_grad():
            out = F.normalize(torch.cat([m(val[i : i + 1]) for i in range(len(val))]), dim=-1)
        print(f"mode {mode}: PyTorch rewrite vs original, cos min = {(out * ref).sum(1).min():.7f}", flush=True)
        with torch.no_grad():
            traced = torch.jit.trace(m, val[:1], strict=False)
        ml = ct.convert(
            traced,
            convert_to="mlprogram",
            inputs=[ct.TensorType(name="image", shape=(1, 3, 336, 336), dtype=np.float32)],
            outputs=[ct.TensorType(name="embedding")],
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS14,
        )
        ml.save(str(here / "models" / f"pe_l14_fast{mode}.mlpackage"))
        print(f"mode {mode}: saved", flush=True)


if __name__ == "__main__":
    main()
