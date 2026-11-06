# PE-Core-L14-336 CoreML benchmark

Exports the PE-Core-L14-336 vision encoder to CoreML, then measures speed and fidelity against PyTorch.
It also includes rewritten graphs for the Apple Neural Engine, quantised variants and hardware profilers.

Measured results and conclusions are in [RESULTS.md](RESULTS.md).

## Run it

```bash
cd apps/pe/coreml_bench
bash setup.sh
bash run.sh --quick
bash run.sh
```

`--quick` checks that everything works. Its timings are not comparable, so use the full run for real numbers.

**Needs:** an Apple Silicon Mac, internet access for `setup.sh`, and about 12 GB of free disk space. The scripts
create their own Python 3.11 environment in `.venv` from `requirements.lock`, and use `core/` from this repo.

**For fair numbers:** plug in, turn off Low Power Mode, and close other heavy apps. Fanless Macs such as the
MacBook Air can throttle during a long run. The drift check at the end of the full run shows if that happened.

## What you get

- `results/summary.md`: latency, sustained throughput and fidelity, each with the speedup against the M2 Pro.
- `results/placement_*.txt`: which device CoreML plans to use for each op, and its estimated cost.
- `pe_coreml_results_<chip>.zip`: all the results, ready to share.

Results marked `(!)` logged a failed Neural Engine compile. CoreML carries on after one, so these runs finish
normally but may not be running purely on the Neural Engine.

If a config fails or times out it is listed in the summary and the run continues; details are in
`results/log.txt`. Re-running `run.sh` reuses models that have already been built, and moves earlier results to
`results_previous/`.

## Hardware profiling

These need no sudo, and read models from `models/`, so build those first with `run.sh`.

```bash
.venv/bin/python profile_hw.py pe_l14_fastC CPU_AND_NE 15
```

```bash
.venv/bin/python profile_hw.py pe_l14_fastC CPU_AND_NE contention
```

```bash
.venv/bin/python micro_ane.py sweep
```

- `profile_hw.py <stem> <unit> <seconds>`: memory traffic per image, and a histogram of Neural Engine bandwidth.
- `profile_hw.py <stem> <unit> contention`: repeats the model under CPU load, with DRAM-heavy and cache-only variants.
- `micro_ane.py [grid|sweep|mlp]`: Neural Engine throughput on stacks of 1x1 convs, with call overhead included, the on-chip working-set limit, and
  whether splitting a wide layer helps.

They use `aneopt.ioreport`, which reads a private Apple API, so channel names may differ on other chips. Only an
M2 Pro on macOS 27 has been tested. For Neural Engine power, use `sudo powermetrics --samplers ane_power`.

The reusable parts of this benchmark are in the [`aneopt`](../../aneopt) package, which `setup.sh` installs. To apply the same
method to another model, see [`OPTIMISATION.md`](../../aneopt/OPTIMISATION.md).

## Scripts

| Script | What it does |
|---|---|
| `export_coreml.py` | Converts the original model to FP16 and FP32 CoreML packages |
| `pe_fast.py` | Builds the rewritten graphs, modes A, B, C and D |
| `quantize.py` | Builds `w8`, `p6`, `p4` and `a8w8` from the FP16 package |
| `variants.py` | Builds quantised mode C, batch-4 graphs and macOS 15 grouped palettes |
| `prep_data.py` | Preprocesses Imagenette into `data/` |
| `bench.py` | Measures latency and saves embeddings for one config |
| `throughput.py` | Measures sustained images/s for one or more workers running together |
| `fidelity.py` | Scores saved embeddings against PyTorch FP32 on CPU |
| `profile_plan.py` | Reports CoreML's per-op device plan and estimated cost |
| `run_all.py`, `summary.py` | Run everything and write the summary |
| `export_weights.py`, `pe_aneforge.py` | Optional: run modes C and D through ANEForge, in a separate environment |

`run.sh` builds and runs `w8` and `p6`. To build the `p4` and `a8w8` packages, run `quantize.py` with those names.
Both gave broken output when tested, as `RESULTS.md` describes.

## Models

| Name | What it is |
|---|---|
| `pe_l14_fp16`, `pe_l14_fp32` | Direct conversion of the original model |
| `pe_l14_fastA` | Original layout, with attention and RoPE cleaned up |
| `pe_l14_fastB` | Channels-first `(B,C,1,S)`, 1x1 convs, RoPE swap as a fixed grouped conv |
| `pe_l14_fastC` | B, with attention run per head |
| `pe_l14_fastD` | C, with wide layers split by channel and a two-stage patch conv. The fastest lossless graph on the M2 Pro |
| `*_w8`, `*_p6` | 8-bit weights, 6-bit palettised weights |
| `*_w8c` | 8-bit conv weights only, which keeps positional tables at full precision |
| `*15_p6g` | 6-bit palette per channel group, needs macOS 15 |
| `*_b4` | Batch size 4 graphs |

The rewrites use the original weights unchanged (see `pe_fast.py`). Fidelity is scored on 100 Imagenette
images against PyTorch FP32 on CPU. Imagenette's 10 classes are easy, so top-1 alone won't reveal small
errors: compare the cosine column too.

## Cleaning up

Delete `.venv`, `.uv`, `data`, `models`, `embeds` and `results`. The weights are also cached in
`~/.cache/huggingface/hub/models--facebook--PE-Core-L14-336`.
