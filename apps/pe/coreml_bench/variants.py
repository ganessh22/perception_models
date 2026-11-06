"""Second-round models: mode C quantised, batch-4 graphs, and macOS 15 per-channel-group palettes."""
import sys
from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as opt
import numpy as np
import torch

from aneopt.export import quantize_weights
from pe_fast import FastPE, pe

MODELS = Path(__file__).parent / "models"


def convert(module, name, batch=1, target=ct.target.macOS14):
    example = torch.rand(batch, 3, 336, 336) * 2 - 1
    with torch.no_grad():
        traced = torch.jit.trace(module, example, strict=False)
    ml = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="image", shape=example.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="embedding")],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=target,
    )
    ml.save(str(MODELS / f"{name}.mlpackage"))
    print("saved", name, flush=True)
    return ml


def optimise(fn, cfg, src, name):
    src_model = ct.models.MLModel(str(MODELS / f"{src}.mlpackage"), skip_model_load=True)
    fn(src_model, opt.OptimizationConfig(global_config=cfg)).save(str(MODELS / f"{name}.mlpackage"))
    print("saved", name, flush=True)


model = pe.VisionTransformer.from_config("PE-Core-L14-336", pretrained=True).eval()
fast_c = FastPE(model, "C").eval()

convert(model, "pe_l14_fp16_b4", batch=4)
convert(fast_c, "pe_l14_fastC_b4", batch=4)
convert(fast_c, "pe_l14_fastC15", target=ct.target.macOS15)

w8 = opt.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
optimise(opt.linear_quantize_weights, w8, "pe_l14_fastC", "pe_l14_fastC_w8")
quantize_weights(MODELS / "pe_l14_fastD.mlpackage", MODELS / "pe_l14_fastD_w8.mlpackage")
quantize_weights(MODELS / "pe_l14_fastD.mlpackage", MODELS / "pe_l14_fastD_w8c.mlpackage", conv_only=True)
optimise(opt.palettize_weights, opt.OpPalettizerConfig(mode="kmeans", nbits=6), "pe_l14_fastC", "pe_l14_fastC_p6")
for bits in (6,):
    cfg = opt.OpPalettizerConfig(mode="kmeans", nbits=bits, granularity="per_grouped_channel", group_size=16)
    optimise(opt.palettize_weights, cfg, "pe_l14_fastC15", f"pe_l14_fastC15_p{bits}g")
