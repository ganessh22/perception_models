"""Memory traffic and Neural Engine bandwidth histogram for one CoreML model, optionally under memory contention.

  profile_hw.py <stem> <compute unit> [seconds]
  profile_hw.py <stem> <compute unit> contention
"""
import json
import sys
import threading
import time
from pathlib import Path

import coremltools as ct
import numpy as np

from aneopt.ioreport import Sampler
from aneopt.measure import GROUPS, measure

HERE = Path(__file__).parent


def stress(threads, buffer_bytes, stop):
    """Threads copying buffers: large ones force DRAM traffic, small ones stay in the CPU cache."""
    def work():
        a = np.ones(buffer_bytes, dtype=np.uint8)
        b = np.zeros(buffer_bytes, dtype=np.uint8)
        while not stop.is_set():
            np.copyto(b, a)
    ts = [threading.Thread(target=work, daemon=True) for _ in range(threads)]
    for t in ts:
        t.start()
    return ts


if __name__ == "__main__":
    stem, unit = sys.argv[1], sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else "15"
    t = time.perf_counter()
    model = ct.models.MLModel(str(HERE / "models" / f"{stem}.mlpackage"), compute_units=getattr(ct.ComputeUnit, unit))
    load_s = round(time.perf_counter() - t, 1)
    x = np.load(HERE / "data" / "val.npy", mmap_mode="r")
    samples = [np.asarray(x[i])[None].astype(np.float32) for i in range(8)]
    state = {"i": 0}

    def predict():
        model.predict({"image": samples[state["i"] % 8]})
        state["i"] += 1

    for _ in range(3):
        predict()
    sampler = Sampler(GROUPS)
    if mode == "contention":
        for threads, size, label in ((0, 0, "none"), (2, 512 << 20, "dram"), (4, 512 << 20, "dram"), (6, 512 << 20, "dram"),
                                     (2, 1 << 18, "cache"), (4, 1 << 18, "cache"), (6, 1 << 18, "cache"), (0, 0, "none")):
            stop = threading.Event()
            stress(threads, size, stop) if threads else None
            time.sleep(1.5)
            r = measure(predict, sampler, 10)
            stop.set()
            time.sleep(1)
            print(json.dumps(dict(stem=stem, threads=threads, load=label, **{k: r[k] for k in ("median_ms", "ane_port_gb_s", "ane_dram_gb_s", "all_dram_gb_s", "cpu_dram_gb_s")}, hist=r["ane_hist"])), flush=True)
    else:
        print(json.dumps(dict(stem=stem, unit=unit, load_s=load_s, **measure(predict, sampler, float(mode)))), flush=True)
