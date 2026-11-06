"""Sustained images/s over a fixed window, with one or more models running at once.

  throughput.py [--secs 30] stem:UNIT:batch [stem:UNIT:batch ...]

One process per worker, so nothing depends on whether coremltools releases the GIL. Each worker loads and
warms up, then all start together. Also checks the first batch's embeddings against the FP32 reference.
"""
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def worker(spec, secs, barrier, queue):
    import coremltools as ct

    stem, unit, batch = spec.split(":")
    batch = int(batch)
    model = ct.models.MLModel(str(HERE / "models" / f"{stem}.mlpackage"), compute_units=getattr(ct.ComputeUnit, unit))
    val = np.load(HERE / "data" / "val.npy", mmap_mode="r")
    x = np.array(val[: batch])
    for _ in range(3):
        out = model.predict({"image": x})["embedding"]

    ref = np.load(HERE / "embeds" / "torch_fp32_cpu.npy")[: batch]
    norm = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
    cos = float((norm(out) * norm(ref)).sum(1).min())

    barrier.wait()
    start = time.perf_counter()
    done = 0
    while time.perf_counter() - start < secs:
        model.predict({"image": x})
        done += batch
    queue.put(dict(spec=spec, images_per_s=round(done / (time.perf_counter() - start), 2), cos_min=round(cos, 6)))


def main():
    args = sys.argv[1:]
    secs = 30
    if args[0] == "--secs":
        secs, args = int(args[1]), args[2:]
    ctx = mp.get_context("spawn")
    barrier, queue = ctx.Barrier(len(args)), ctx.Queue()
    procs = [ctx.Process(target=worker, args=(s, secs, barrier, queue)) for s in args]
    for p in procs:
        p.start()
    results = [queue.get() for _ in procs]
    for p in procs:
        p.join()
    results.sort(key=lambda r: args.index(r["spec"]))  # completion order varies; keep the order they were asked for
    line = json.dumps(dict(total_images_per_s=round(sum(r["images_per_s"] for r in results), 2), workers=results))
    with open(HERE / "throughput.jsonl", "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


if __name__ == "__main__":
    main()
