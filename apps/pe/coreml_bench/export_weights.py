"""Dump PE-Core-L14-336 weights to one npz for pe_aneforge.py: `export_weights.py <out.npz>`, in the benchmark venv."""
import sys

import numpy as np
import torch

import core.vision_encoder.pe as pe

m = pe.VisionTransformer.from_config("PE-Core-L14-336", pretrained=True).eval()
m.rope.update_grid(torch.device("cpu"), 24, 24)
ap = m.attn_pool
d = {
    "conv1_w": m.conv1.weight, "cls": m.class_embedding, "pos": m.positional_embedding,
    "freq": m.rope.freq[0], "ln_pre_w": m.ln_pre.weight, "ln_pre_b": m.ln_pre.bias,
    "ln_post_w": m.ln_post.weight, "ln_post_b": m.ln_post.bias, "final": m.proj,
    "probe": ap.probe, "pool_in_w": ap.attn.in_proj_weight, "pool_in_b": ap.attn.in_proj_bias,
    "pool_out_w": ap.attn.out_proj.weight, "pool_out_b": ap.attn.out_proj.bias,
    "pool_ln_w": ap.layernorm.weight, "pool_ln_b": ap.layernorm.bias,
    "pool_fc_w": ap.mlp.c_fc.weight, "pool_fc_b": ap.mlp.c_fc.bias,
    "pool_proj_w": ap.mlp.c_proj.weight, "pool_proj_b": ap.mlp.c_proj.bias,
}
if m.conv1.bias is not None:
    d["conv1_b"] = m.conv1.bias
for i, r in enumerate(m.transformer.resblocks):
    d.update({
        f"{i}_ln1_w": r.ln_1.weight, f"{i}_ln1_b": r.ln_1.bias, f"{i}_ln2_w": r.ln_2.weight, f"{i}_ln2_b": r.ln_2.bias,
        f"{i}_in_w": r.attn.in_proj_weight, f"{i}_in_b": r.attn.in_proj_bias,
        f"{i}_out_w": r.attn.out_proj.weight, f"{i}_out_b": r.attn.out_proj.bias,
        f"{i}_fc_w": r.mlp.c_fc.weight, f"{i}_fc_b": r.mlp.c_fc.bias,
        f"{i}_proj_w": r.mlp.c_proj.weight, f"{i}_proj_b": r.mlp.c_proj.bias,
    })
a = m.transformer.resblocks[0].attn
print("eps", m.ln_pre.eps, "heads", a.num_heads, "head_dim", a.head_dim, "scale", a.scale, "pool heads", ap.num_heads, "conv1", m.conv1)
print("probe", ap.probe.shape, "freq", d["freq"].shape, "final", m.proj.shape, "layers", len(m.transformer.resblocks))
np.savez(sys.argv[1], **{k: v.detach().numpy().astype(np.float32) for k, v in d.items()})
