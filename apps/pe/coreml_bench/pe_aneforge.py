"""PE-Core-L14-336 on the Neural Engine through ANEForge, mirroring modes C and D in pe_fast.py.

  pe_aneforge.py <weights.npz> <bench dir> <C|D> [--layers N] [--compress int8] [--spelled-ln] [--eval K]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import aneforge as af
from aneforge.graph import _const

ap = argparse.ArgumentParser()
ap.add_argument("weights")
ap.add_argument("bench")
ap.add_argument("mode")
ap.add_argument("--build-dir", default=str(Path(__file__).parent / "build"))
ap.add_argument("--layers", type=int, default=24)
ap.add_argument("--compress")
ap.add_argument("--spelled-ln", action="store_true")
ap.add_argument("--eval", type=int, default=0)
ap.add_argument("--seconds", type=float, default=12)
args = ap.parse_args()

bench = Path(args.bench)
sys.path.insert(0, str(bench / "../../aneopt/src/aneopt"))
from ioreport import Sampler  # noqa: E402  (only needs ctypes)

w = np.load(args.weights)
E, H, D, EPS, S = 1024, 16, 64, 1e-5, 577
split_all = args.mode == "D"


def lin(x, W, b=None):
    return af.conv(x, W[:, :, None, None], bias=b)


def ln(x, g, b):
    if not args.spelled_ln:
        return x.channel_layer_norm(g, b, EPS)
    z = x - x.mean((1,))
    var = (z * z).mean((1,))
    return z * (var.adds(EPS).rsqrt()) * g[None, :, None, None] + b[None, :, None, None]


def swap(x):
    weight = np.tile(np.array([[0.0, -1.0], [1.0, 0.0]], np.float32), (E // 2, 1))[:, :, None, None]
    return af.conv(x, weight, groups=E // 2)


def attend(q, k, v, scale):
    s = (k.transpose([0, 1, 3, 2]) @ q) * scale
    return v @ s.softmax(-2)


freq = w["freq"]
table = lambda t: np.tile(t.T, (H, 1))[None, :, None, :]
COS, SIN = table(np.cos(freq)), table(np.sin(freq))


def block(x, i):
    g = lambda n: w[f"{i}_{n}"]
    y = ln(x, g("ln1_w"), g("ln1_b"))
    Wi, bi = g("in_w"), g("in_b")
    if split_all:
        q, k, v = [lin(y, Wi[j * E:(j + 1) * E], bi[j * E:(j + 1) * E]) for j in range(3)]
    else:
        q, k, v = af.split(lin(y, Wi, bi), 3, axis=1)
    q = q * COS + swap(q) * SIN
    k = k * COS + swap(k) * SIN
    parts = zip(*[af.split(t.reshape(1, H, D, S), H, axis=1) for t in (q, k, v)])
    a = af.concat([attend(qi, ki, vi, D ** -0.5) for qi, ki, vi in parts], axis=1)
    x = x + lin(a.reshape(1, E, 1, S), g("out_w"), g("out_b"))
    y = ln(x, g("ln2_w"), g("ln2_b"))
    Wf, bf, Wp, bp = g("fc_w"), g("fc_b"), g("proj_w"), g("proj_b")
    if split_all:
        out = None
        for j in range(4):
            t = lin(lin(y, Wf[j * E:(j + 1) * E], bf[j * E:(j + 1) * E]).gelu(), Wp[:, j * E:(j + 1) * E])
            out = t if out is None else out + t
        return x + (out + bp[None, :, None, None])
    return x + lin(lin(y, Wf, bf).gelu(), Wp, bp)


img = af.input((1, 3, 336, 336))
# space_to_depth by 7 or 14 fails to compile here, and a 14x14 conv is rejected by a shape check. So patch embedding is an
# exact two-stage conv: identity 7x7 kernels (stride 7) to unshuffle pixels, then the original weights as 2x2 stride 2.
ident = np.zeros((147, 3, 7, 7), np.float32)
w2 = np.zeros((E, 147, 2, 2), np.float32)
for c in range(3):
    for yy in range(7):
        for xx in range(7):
            ch = c * 49 + yy * 7 + xx
            ident[ch, c, yy, xx] = 1
            for Y in range(2):
                for X in range(2):
                    w2[:, ch, Y, X] = w["conv1_w"][:, c, 7 * Y + yy, 7 * X + xx]
h = af.conv(af.conv(img, ident, stride=7), w2, stride=2).reshape(1, E, 1, 576)
x = af.concat([_const(w["cls"][None, :, None, None]), h], axis=3) + w["pos"].T[None, :, None, :]
x = ln(x, w["ln_pre_w"], w["ln_pre_b"])
for i in range(args.layers):
    x = block(x, i)
x = ln(x, w["ln_post_w"], w["ln_post_b"])

Wp, bp = w["pool_in_w"], w["pool_in_b"]
PH, PD = 8, 128
pq = (w["probe"][0, 0] @ Wp[:E].T + bp[:E]).reshape(1, PH, PD, 1)
k = lin(x, Wp[E:2 * E], bp[E:2 * E]).reshape(1, PH, PD, S)
v = lin(x, Wp[2 * E:], bp[2 * E:]).reshape(1, PH, PD, S)
a = attend(_const(pq), k, v, PD ** -0.5)
y = lin(a.reshape(1, E, 1, 1), w["pool_out_w"], w["pool_out_b"])
y = y + lin(lin(ln(y, w["pool_ln_w"], w["pool_ln_b"]), w["pool_fc_w"], w["pool_fc_b"]).gelu(), w["pool_proj_w"], w["pool_proj_b"])
out = lin(y, w["final"].T).reshape(1, 1024)

t0 = time.perf_counter()
kw = dict(compress=args.compress, opt=0) if args.compress else dict(opt=0)
kw["build_dir"] = args.build_dir
net = af.compile(out, **kw)
compile_s = time.perf_counter() - t0
print(f"compiled in {compile_s:.1f} s, type {type(net).__name__}, ops {getattr(net, 'n_ops', '?')}", flush=True)

val = np.load(bench / "data" / "val.npy", mmap_mode="r")
t0 = time.perf_counter()
net(np.array(val[:1]))
first_ms = (time.perf_counter() - t0) * 1000
for i in range(5):
    net(np.array(val[i:i + 1]))
lat = []
for i in range(30):
    x_ = np.array(val[i:i + 1])
    t0 = time.perf_counter()
    net(x_)
    lat.append(time.perf_counter() - t0)

# Zero-copy path: binding and copying excluded, so this is the program itself plus dispatch.
view = net.input_view()
view[...] = np.array(val[:1]).astype(np.float16)
raw = []
for _ in range(30):
    t0 = time.perf_counter()
    net.execute()
    raw.append(time.perf_counter() - t0)

GROUPS = (("AMC Stats", "Perf Counters"), ("PMP", "AF BW"))
amc = lambda n: ("AMC Stats", "Perf Counters", n)
sampler = Sampler(GROUPS)
start = sampler.snapshot()
t0 = time.perf_counter()
n = 0
while time.perf_counter() - t0 < args.seconds:
    net.execute()
    n += 1
delta = sampler.delta(start, sampler.snapshot())
dt = time.perf_counter() - t0
gb = lambda *names: sum(delta.get(amc(y), 0) for y in names) / 1e9

rec = dict(mode=args.mode, layers=args.layers, compress=args.compress, spelled_ln=args.spelled_ln, compile_s=round(compile_s, 1), first_ms=round(first_ms, 1),
           median_ms=round(float(np.median(lat)) * 1000, 1), execute_ms=round(float(np.median(raw)) * 1000, 1), images=n,
           ane_read_gb_img=round(gb("DIE0 ANE RD") / n, 3), ane_write_gb_img=round(gb("DIE0 ANE WR") / n, 3), ane_dram_read_gb_img=round(gb("DIE0 ANE DCS RD") / n, 3),
           ane_port_gb_s=round(gb("DIE0 ANE RD", "DIE0 ANE WR") / dt, 1))
if args.eval:
    idx = np.linspace(0, len(val) - 1, args.eval).astype(int)
    tag = f"aneforge_{args.mode}{'_' + args.compress if args.compress else ''}{'_spelled' if args.spelled_ln else ''}"
    np.save(bench / "embeds" / f"{tag}.npy", np.stack([net(np.array(val[i:i + 1])).reshape(-1) for i in idx]))
print(json.dumps(rec), flush=True)
net.release()
