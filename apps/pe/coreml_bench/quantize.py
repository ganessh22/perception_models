"""Build quantised variants of the FP16 PE-Core-L14 CoreML model."""
import sys
import time
from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as opt
import numpy as np

HERE = Path(__file__).parent
MODELS = HERE / "models"
SRC = str(MODELS / "pe_l14_fp16.mlpackage")


def base(load=False):
    return ct.models.MLModel(SRC, skip_model_load=not load, compute_units=ct.ComputeUnit.CPU_ONLY)


def weights_int8(model):
    cfg = opt.OptimizationConfig(global_config=opt.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8"))
    return opt.linear_quantize_weights(model, cfg)


def palette(nbits):
    cfg = opt.OptimizationConfig(global_config=opt.OpPalettizerConfig(mode="kmeans", nbits=nbits))
    return opt.palettize_weights(base(), cfg)


def a8w8():
    calib = np.load(HERE / "data" / "calib.npy")
    samples = [{"image": calib[i : i + 1]} for i in range(len(calib))]
    cfg = opt.OptimizationConfig(global_config=opt.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8"))
    return weights_int8(opt.linear_quantize_activations(base(load=True), cfg, sample_data=samples))


VARIANTS = {
    "w8": lambda: weights_int8(base()),
    "p6": lambda: palette(6),
    "p4": lambda: palette(4),
    "a8w8": a8w8,
}

for name in sys.argv[1:] or VARIANTS:
    t0 = time.time()
    VARIANTS[name]().save(str(MODELS / f"pe_l14_{name}.mlpackage"))
    print(f"{name}: built in {time.time() - t0:.0f}s", flush=True)
