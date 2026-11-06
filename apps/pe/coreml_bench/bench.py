"""Latency and fidelity benchmark, one config per process so memory and caches don't leak across runs.

  bench.py run torch  <fp16|fp32> <cpu|mps> [--n N] [--eval K]
  bench.py run coreml <package-stem>  <CPU_ONLY|CPU_AND_GPU|CPU_AND_NE|ALL> [--n N] [--eval K]

Appends one JSON line to results.jsonl and saves embeddings to embeds/<tag>.npy when --eval K > 0.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "data"
(HERE / "embeds").mkdir(exist_ok=True)


def run_torch(precision, device):
    import torch

    import core.vision_encoder.pe as pe

    t0 = time.perf_counter()
    model = pe.VisionTransformer.from_config("PE-Core-L14-336", pretrained=True).eval().to(device)
    dtype = torch.float32
    if precision == "fp16":
        dtype = torch.float16
        model = model.half()
    # The RoPE table lives on a plain (non-Module) object, so .half() misses it. Build it, then
    # convert it, otherwise every attention layer silently promotes to FP32 and casts back.
    model.rope.update_grid(torch.device(device), 24, 24)
    model.rope.freq = model.rope.freq.to(dtype)
    load_s = time.perf_counter() - t0

    def predict(x):
        with torch.inference_mode():
            t = torch.from_numpy(x).to(device).to(dtype)
            return model(t).float().cpu().numpy()  # .cpu() forces MPS to finish

    return predict, load_s


def run_coreml(stem, unit):
    import coremltools as ct

    t0 = time.perf_counter()
    model = ct.models.MLModel(
        str(HERE / "models" / f"{stem}.mlpackage"), compute_units=getattr(ct.ComputeUnit, unit)
    )
    load_s = time.perf_counter() - t0
    return (lambda x: model.predict({"image": x})["embedding"]), load_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backend", choices=["torch", "coreml"])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--eval", type=int, default=0)
    args = ap.parse_args()

    predict, load_s = (run_torch if args.backend == "torch" else run_coreml)(args.a, args.b)
    val = np.load(DATA / "val.npy", mmap_mode="r")

    t0 = time.perf_counter()
    predict(np.array(val[:1]))
    first_s = time.perf_counter() - t0
    for i in range(args.warmup):
        predict(np.array(val[i : i + 1]))

    times = []
    for i in range(args.n):
        x = np.array(val[i : i + 1])
        t0 = time.perf_counter()
        predict(x)
        times.append(time.perf_counter() - t0)
    times = np.array(times) * 1000

    tag = f"{args.backend}_{args.a}_{args.b}"
    if args.eval:
        # val is sorted by class, so sample evenly rather than taking the first K
        idx = np.linspace(0, len(val) - 1, args.eval).astype(int)
        out = [predict(np.array(val[i : i + 1]))[0] for i in idx]
        np.save(HERE / "embeds" / f"{tag}.npy", np.stack(out))

    rec = dict(
        tag=tag,
        load_s=round(load_s, 2),
        first_ms=round(first_s * 1000, 1),
        median_ms=round(float(np.median(times)), 1),
        p90_ms=round(float(np.percentile(times, 90)), 1),
        min_ms=round(float(times.min()), 1),
        fps=round(1000 / float(np.median(times)), 2),
    )
    with open(HERE / "results.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
