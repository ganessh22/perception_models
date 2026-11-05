"""Memory traffic and Neural Engine bandwidth histogram for one CoreML model, optionally under memory contention.

  profile_hw.py <stem> <compute unit> [seconds]
  profile_hw.py <stem> <compute unit> contention
"""
import json
import re
import sys
import threading
import time
from pathlib import Path

import coremltools as ct
import numpy as np

from ioreport import Sampler

HERE = Path(__file__).parent
GROUPS = (("AMC Stats", "Perf Counters"), ("SoC Stats", "Cluster Power States"), ("PMP", "AF BW"), ("PMP", "DCS Floor"), ("PMP", "FAB Floor"))
amc = lambda name: ("AMC Stats", "Perf Counters", name)


def hist_stats(h):
    """Bucket labels look like '  8GB/s'. Returns the mean, percentiles and top bucket, weighted by time."""
    pts = [(float(re.match(r"\s*([\d.]+)", k).group(1)), v) for k, v in h.items() if re.match(r"\s*[\d.]+", k) and v]
    if not pts:
        return {}
    pts.sort()
    total = sum(v for _, v in pts)
    cum, out = 0, {}
    for b, v in pts:
        cum += v
        for q, name in ((0.5, "p50"), (0.9, "p90"), (0.99, "p99")):
            if name not in out and cum / total >= q:
                out[name] = b
    out["max_bucket"] = pts[-1][0]
    out["mean_gb_s"] = round(sum(b * v for b, v in pts) / total, 1)
    out["over_30_gb_s"] = round(sum(v for b, v in pts if b > 30) / total, 3)
    return out


def measure(predict, sampler, seconds):
    start = sampler.snapshot()
    t0 = time.perf_counter()
    lat = []
    while time.perf_counter() - t0 < seconds:
        a = time.perf_counter()
        predict()
        lat.append(time.perf_counter() - a)
    busy = sampler.delta(start, sampler.snapshot())
    dt = time.perf_counter() - t0
    n = len(lat)
    gb = lambda *names: round(sum(busy.get(amc(x), 0) for x in names) / 1e9, 3)
    return dict(
        images=n,
        median_ms=round(float(np.median(lat)) * 1000, 1),
        ane_read_gb_img=round(gb("DIE0 ANE RD") / n, 3),
        ane_write_gb_img=round(gb("DIE0 ANE WR") / n, 3),
        ane_dram_read_gb_img=round(gb("DIE0 ANE DCS RD") / n, 3),
        ane_dram_write_gb_img=round(gb("DIE0 ANE DCS WR") / n, 3),
        ane_port_gb_s=round(gb("DIE0 ANE RD", "DIE0 ANE WR") / dt, 1),
        ane_dram_gb_s=round(gb("DIE0 ANE DCS RD", "DIE0 ANE DCS WR") / dt, 1),
        all_dram_gb_s=round(gb("DCS") / dt, 1),
        cpu_dram_gb_s=round(gb(*[f"DIE0 {c} DCS {d}" for c in ("PCPU0", "PCPU1", "ECPU0") for d in ("RD", "WR")]) / dt, 1),
        gpu_dram_gb_s=round(gb("DIE0 GFX DCS RD", "DIE0 GFX DCS WR") / dt, 1),
        ane_hist=hist_stats(busy.get(("PMP", "AF BW", "ANE0 RD+WR"), {})),
    )


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
