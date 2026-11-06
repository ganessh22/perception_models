"""Measure what the Neural Engine can do on stacks of 1x1 convs, varying activation size and weight size.

Shape (1, C, 1, S): activations are 2*C*S bytes, weights 2*C*C bytes per layer, and each layer does C*C*S multiply-adds.
"""
import json
import shutil
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from ioreport import Sampler
from profile_hw import GROUPS, amc, hist_stats

HERE = Path(__file__).parent
OUT = HERE / "micro"
OUT.mkdir(exist_ok=True)
# channels, tokens, layers
GRID = [(1024, 16, 96), (1024, 64, 48), (1024, 144, 24), (1024, 288, 16), (1024, 577, 8),
        (512, 577, 24), (256, 577, 64), (256, 2304, 24), (128, 4608, 32), (64, 9216, 64)]


class Stack(torch.nn.Module):
    def __init__(self, c, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(torch.nn.Conv2d(c, c, 1, bias=False) for _ in range(layers))
        torch.manual_seed(0)
        for layer in self.layers:
            torch.nn.init.normal_(layer.weight, std=1 / c**0.5)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x)) + x * 0.5
        return x


rows = []
for c, s, k in GRID:
    path = OUT / f"c{c}_s{s}_k{k}.mlpackage"
    x = torch.rand(1, c, 1, s)
    if not path.exists():
        traced = torch.jit.trace(Stack(c, k).eval(), x)
        ct.convert(traced, inputs=[ct.TensorType(name="x", shape=x.shape)], convert_to="mlprogram",
                   compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS14).save(str(path))
    model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    data = {"x": x.numpy()}
    for _ in range(5):
        model.predict(data)
    sampler = Sampler(GROUPS)
    start = sampler.snapshot()
    t0 = time.perf_counter()
    lat = []
    while time.perf_counter() - t0 < 6:
        a = time.perf_counter()
        model.predict(data)
        lat.append(time.perf_counter() - a)
    delta = sampler.delta(start, sampler.snapshot())
    dt = time.perf_counter() - t0
    n = len(lat)
    gb = lambda *names: sum(delta.get(amc(y), 0) for y in names) / 1e9
    ms = float(np.median(lat)) * 1000
    macs = c * c * s * k
    rows.append(dict(c=c, s=s, k=k, gmac=round(macs / 1e9, 2), activ_kb=2 * c * s // 1024, weights_mb=round(2 * c * c * k / 1e6, 1),
                     ms=round(ms, 1), tmac_s=round(macs / ms / 1e9, 2),
                     port_gb_s=round(gb("DIE0 ANE RD", "DIE0 ANE WR") / dt, 1), dram_gb_s=round(gb("DIE0 ANE DCS RD", "DIE0 ANE DCS WR") / dt, 1),
                     read_gb=round(gb("DIE0 ANE RD") / n, 2), dram_read_gb=round(gb("DIE0 ANE DCS RD") / n, 2), hist=hist_stats(delta.get(("PMP", "AF BW", "ANE0 RD+WR"), {}))))
    print(json.dumps(rows[-1]), flush=True)
    del model
shutil.rmtree(OUT, ignore_errors=True)
