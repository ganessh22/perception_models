"""Measure what this Neural Engine can do on synthetic graphs, varying activation size and weight size.

  grid   stacks of 1x1 convs across activation and weight sizes
  sweep  stacks of 1x1 convs with activations from about 0.5 to 8 MB, to find the on-chip working-set limit
  mlp    an MLP-shaped stack with a 4x hidden layer, as one wide conv and split by channel

Shape (1, C, 1, S): activations are 2*C*S bytes, weights 2*C*C bytes per layer, and each layer does C*C*S multiply-adds.
"""
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from .ioreport import Sampler
from .measure import GROUPS, amc, hist_stats

# channels, tokens, layers
GRID = [(1024, 16, 96), (1024, 64, 48), (1024, 144, 24), (1024, 288, 16), (1024, 577, 8),
        (512, 577, 24), (256, 577, 64), (256, 2304, 24), (128, 4608, 32), (64, 9216, 64)]
SWEEP = [(256, s, 16) for s in (1024, 2048, 3072, 3584, 4096, 5120, 6144, 7168, 8192, 9216, 10240, 12288, 16384)] + [
    (1024, s, 8) for s in (577, 1024, 1536, 2048, 2560, 3072, 4096)]


def _stack(c, layers):
    import torch

    class Stack(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList(torch.nn.Conv2d(c, c, 1, bias=False) for _ in range(layers))
            torch.manual_seed(0)
            for layer in self.layers:
                torch.nn.init.normal_(layer.weight, std=1 / c**0.5)

        def forward(self, x):
            for layer in self.layers:
                x = torch.relu(layer(x)) + x * 0.5
            return x

    return Stack()


def _mlp(c, layers, split):
    import torch

    class Mlp(torch.nn.Module):
        """x + proj(gelu(fc(x))) with a hidden layer of 4C channels, either as one wide conv or as four C-channel branches."""

        def __init__(self):
            super().__init__()
            n = 4 if split else 1
            conv = lambda a, b: torch.nn.Conv2d(a, b, 1, bias=False)
            self.fc = torch.nn.ModuleList(torch.nn.ModuleList(conv(c, 4 * c // n) for _ in range(n)) for _ in range(layers))
            self.proj = torch.nn.ModuleList(torch.nn.ModuleList(conv(4 * c // n, c) for _ in range(n)) for _ in range(layers))
            torch.manual_seed(0)
            for group in (*self.fc, *self.proj):
                for layer in group:
                    torch.nn.init.normal_(layer.weight, std=1 / c**0.5)

        def forward(self, x):
            for fc, proj in zip(self.fc, self.proj):
                y = 0
                for f, p in zip(fc, proj):
                    y = y + p(torch.nn.functional.gelu(f(x)))
                x = x + y
            return x

    return Mlp()


def _measure(out, name, module, x, macs, seconds, **meta):
    import coremltools as ct
    import torch

    path = Path(out) / f"{name}.mlpackage"
    traced = torch.jit.trace(module.eval(), x)
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
    while time.perf_counter() - t0 < seconds:
        a = time.perf_counter()
        model.predict(data)
        lat.append(time.perf_counter() - a)
    delta = sampler.delta(start, sampler.snapshot())
    dt = time.perf_counter() - t0
    n = len(lat)
    gb = lambda *names: sum(delta.get(amc(y), 0) for y in names) / 1e9
    ms = float(np.median(lat)) * 1000
    row = dict(**meta, gmac=round(macs / 1e9, 2), ms=round(ms, 1), tmac_s=round(macs / ms / 1e9, 2),
               port_gb_s=round(gb("DIE0 ANE RD", "DIE0 ANE WR") / dt, 1), dram_gb_s=round(gb("DIE0 ANE DCS RD", "DIE0 ANE DCS WR") / dt, 1),
               read_gb=round(gb("DIE0 ANE RD") / n, 3), write_gb=round(gb("DIE0 ANE WR") / n, 3), dram_read_gb=round(gb("DIE0 ANE DCS RD") / n, 3),
               dram_write_gb=round(gb("DIE0 ANE DCS WR") / n, 3), hist=hist_stats(delta.get(("PMP", "AF BW", "ANE0 RD+WR"), {})))
    print(json.dumps(row), flush=True)
    del model
    shutil.rmtree(path, ignore_errors=True)
    return row


def run(mode="sweep", seconds=6, grid=None):
    """Run a probe and return its rows. `grid` overrides the (channels, tokens, layers) list for grid and sweep."""
    import torch

    rows = []
    with tempfile.TemporaryDirectory(prefix="aneopt_") as out:
        if mode == "mlp":
            c, s, k = 1024, 577, 8
            for split in (False, True):
                rows.append(_measure(out, f"mlp_split{int(split)}", _mlp(c, k, split), torch.rand(1, c, 1, s), 2 * c * 4 * c * s * k, seconds,
                                     split=split, c=c, s=s, k=k, weights_mb=round(2 * 2 * c * 4 * c * k / 1e6, 1)))
        else:
            for c, s, k in grid or {"grid": GRID, "sweep": SWEEP}[mode]:
                rows.append(_measure(out, f"c{c}_s{s}_k{k}", _stack(c, k), torch.rand(1, c, 1, s), c * c * s * k, seconds,
                                     c=c, s=s, k=k, activ_kb=2 * c * s // 1024, weights_mb=round(2 * c * c * k / 1e6, 1)))
    return rows


def find_step(rows, drop=0.15):
    """Where MAC/s first falls by more than `drop` as activations grow, using the channel count that brackets it most tightly."""
    steps = []
    for c in {r["c"] for r in rows}:
        series = sorted((r for r in rows if r["c"] == c), key=lambda r: r["activ_kb"])
        for a, b in zip(series, series[1:]):
            if b["tmac_s"] < a["tmac_s"] * (1 - drop):
                steps.append(dict(channels=c, last_good_kb=a["activ_kb"], first_slow_kb=b["activ_kb"], tmac_before=a["tmac_s"], tmac_after=b["tmac_s"]))
                break
    return min(steps, key=lambda s: s["first_slow_kb"] - s["last_good_kb"], default=None)
