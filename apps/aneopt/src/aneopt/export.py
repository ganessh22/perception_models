"""Convert a rewritten PyTorch module to CoreML, and compress its weights.

Compression can break a model without any error, so always compare the result with a reference. See OPTIMISATION.md.
"""
from pathlib import Path


def convert(module, example, path, input_name="input", output_name="output", target=None):
    """Trace `module` on `example` and save an FP16 ML program. `target` is a coremltools target, macOS 14 by default."""
    import coremltools as ct
    import numpy as np
    import torch

    module = module.eval()
    with torch.no_grad():
        traced = torch.jit.trace(module, example, strict=False)
    model = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name=input_name, shape=example.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name=output_name)],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=target or ct.target.macOS14,
    )
    model.save(str(path))
    return model


def _load(path):
    import coremltools as ct

    return ct.models.MLModel(str(Path(path)), skip_model_load=True)


def quantize_weights(src, dst, conv_only=False):
    """8-bit symmetric weights. `conv_only` leaves other constants, such as positional tables, at full precision."""
    import coremltools.optimize.coreml as opt

    op = opt.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
    if conv_only:
        cfg = opt.OptimizationConfig(global_config=None, op_type_configs={"conv": op})
    else:
        cfg = opt.OptimizationConfig(global_config=op)
    model = opt.linear_quantize_weights(_load(src), cfg)
    model.save(str(dst))
    return model


def palettize_weights(src, dst, nbits=6, group_size=None):
    """k-means palette. `group_size` makes one palette per channel group, which needs a package targeting macOS 15."""
    import coremltools.optimize.coreml as opt

    kwargs = dict(granularity="per_grouped_channel", group_size=group_size) if group_size else {}
    cfg = opt.OptimizationConfig(global_config=opt.OpPalettizerConfig(mode="kmeans", nbits=nbits, **kwargs))
    model = opt.palettize_weights(_load(src), cfg)
    model.save(str(dst))
    return model
