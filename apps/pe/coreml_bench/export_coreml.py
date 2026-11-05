"""Export the PE-Core-L14-336 vision encoder to CoreML (FP16 and FP32)."""
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

import core.vision_encoder.pe as pe

OUT = Path(__file__).parent / "models"
OUT.mkdir(exist_ok=True)
SIZE = 336


def main():
    torch.manual_seed(0)
    model = pe.VisionTransformer.from_config("PE-Core-L14-336", pretrained=True).eval()
    print("params (M):", sum(p.numel() for p in model.parameters()) / 1e6)
    example = torch.rand(1, 3, SIZE, SIZE) * 2 - 1
    with torch.no_grad():
        ref = model(example)
        traced = torch.jit.trace(model, example, strict=False)
        assert torch.allclose(traced(example), ref, atol=1e-4), "trace mismatch"
    print("output shape:", ref.shape)

    for name, precision in [("fp16", ct.precision.FLOAT16), ("fp32", ct.precision.FLOAT32)]:
        t0 = time.time()
        ml = ct.convert(
            traced,
            convert_to="mlprogram",
            inputs=[ct.TensorType(name="image", shape=example.shape, dtype=np.float32)],
            outputs=[ct.TensorType(name="embedding")],
            compute_precision=precision,
            minimum_deployment_target=ct.target.macOS14,
        )
        path = OUT / f"pe_l14_{name}.mlpackage"
        ml.save(str(path))
        out = ml.predict({"image": example.numpy()})["embedding"]
        cos = np.dot(out[0], ref[0].numpy()) / (np.linalg.norm(out[0]) * np.linalg.norm(ref[0].numpy()))
        print(f"{name}: convert {time.time() - t0:.0f}s, cosine vs PyTorch FP32 = {cos:.6f}", flush=True)


if __name__ == "__main__":
    main()
