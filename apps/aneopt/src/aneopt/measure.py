"""Latency and Neural Engine memory traffic for any CoreML package, using the IOReport counters."""
import re
import time

import numpy as np

from .ioreport import Sampler

GROUPS = (("AMC Stats", "Perf Counters"), ("SoC Stats", "Cluster Power States"), ("PMP", "AF BW"), ("PMP", "DCS Floor"), ("PMP", "FAB Floor"))


def amc(name):
    return ("AMC Stats", "Perf Counters", name)


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
    """Call predict() repeatedly for `seconds`. Returns median latency, and traffic per call and per second."""
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


def random_inputs(model):
    """Random float32 inputs matching a package's multi-array inputs. Use real data if the model's cost depends on values."""
    inputs = {}
    for desc in model.get_spec().description.input:
        if desc.type.WhichOneof("Type") != "multiArrayType":
            raise ValueError(f"input {desc.name!r} isn't a multi-array, so pass your own inputs to profile_package()")
        inputs[desc.name] = np.random.default_rng(0).random(tuple(desc.type.multiArrayType.shape), dtype=np.float32)
    return inputs


def profile_package(path, unit="CPU_AND_NE", seconds=12, inputs=None, warmup=3):
    """Load a package and measure it. `inputs` is one prediction's input dict, or a list of them to cycle through."""
    import coremltools as ct

    t = time.perf_counter()
    model = ct.models.MLModel(str(path), compute_units=getattr(ct.ComputeUnit, unit))
    load_s = round(time.perf_counter() - t, 1)
    samples = inputs if isinstance(inputs, list) else [inputs or random_inputs(model)]
    state = {"i": 0}

    def predict():
        model.predict(samples[state["i"] % len(samples)])
        state["i"] += 1

    for _ in range(warmup):
        predict()
    return dict(unit=unit, load_s=load_s, **measure(predict, Sampler(GROUPS), seconds))
