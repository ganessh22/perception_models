# PE-Core-L14-336 CoreML benchmark results

Measured in November 2026. The raw data behind every table is in [`reference/`](reference).

## Summary

- **Best single-device choice:** rewritten mode C with 8-bit weights on the Neural Engine. It takes 60.8 ms on the M4,
  2.5x faster than PyTorch FP16 on MPS (152.1 ms), with mean cosine 0.9988 against PyTorch FP32 on CPU.
- **Best fidelity on the Neural Engine:** unquantised mode C at 79.0 ms on the M4, with mean cosine 0.99996.
- **Highest throughput worth using:** 8-bit mode C on the Neural Engine plus FP16 on the GPU, at 24.0 images/s on the M4.
  The 6-bit pair reaches 25.4 images/s but with lower fidelity.
- **M4 vs M2 Pro:** the M4's Neural Engine is 1.18-1.25x faster on models that compile cleanly, and its 10-core GPU is
  about 26% slower than the M2 Pro's 16-core GPU. The best Neural Engine + GPU pair is about equal on both machines.
- **What limits the Neural Engine, on the M2 Pro:** about half of mode C's time looks like moving data through the Neural
  Engine's own memory port, which appears capped near 60 GB/s. Shared DRAM bandwidth is not the limit.
- **Avoid:** 4-bit palettes and activation quantisation, which gave broken output, and the original graph with 8-bit
  weights on the GPU, which takes over 800 ms on both machines.
- **Newer and faster, on the M2 Pro only:** mode D takes 73.5 ms against mode C's 93.2-94.2 ms, with the same fidelity. With
  8-bit conv weights it takes 68.5 ms, with a worst cosine of 0.9983. The tables below predate it, so see
  [mode D](#mode-d-m2-pro-only).

## Test setup

| | M2 Pro | M4 |
|---|---|---|
| Model identifier | Mac14,9 | Mac16,10 |
| GPU cores | 16 | 10 |
| Memory | 16 GB | 16 GB |
| macOS | 27.0.1 | 26.7.2 |

**Software:** Python 3.11 and the versions pinned in [`requirements.lock`](requirements.lock), including coremltools 9.0
and torch 2.7.0. numpy is pinned to 2.1.2 because coremltools 9.0 failed with numpy 2.4.

**Method:**
- **Latency:** median of 30 single-image predictions after warm-up, one config at a time.
- **Throughput:** 30 seconds of sustained prediction. With several workers, each runs in its own process and they
  start together.
- **Fidelity:** cosine similarity against PyTorch FP32 on CPU, and zero-shot top-1, on 100 images sampled evenly from
  500 Imagenette validation images.
- **Quantisation:** activation quantisation was calibrated on 80 Imagenette training images.

**Reading the numbers:**
- The two Macs run different macOS versions, so some differences could come from the OS rather than the chip.
- Imagenette has only 10 easy classes, so top-1 is close to 100% for any usable model. Use the cosine columns.
- The M2 Pro numbers are the first measurement of each config. Repeats of three configs are in the raw data, and one of
  them differs noticeably: FP16 on the GPU measured 96.8 ms and then 106.6 ms.

## Latency, batch size 1

`(!)` means the Neural Engine compile logged a failure, as described under [compile failures](#neural-engine-compile-failures-on-the-m4).
Sorted by M4 latency.

| Model | Device | M2 Pro ms | M4 ms | M4 speedup |
|---|---|---:|---:|---:|
| fastC_p6 | Neural Engine | 69.1 | 55.2 | 1.25x |
| fastC15_p6g | Neural Engine | 68.3 | 55.4 | 1.23x |
| fastC_w8 | Neural Engine | 75.7 | 60.8 | 1.25x |
| fastC | Neural Engine | 93.2 | 79.0 | 1.18x |
| fastC15 | Neural Engine | 95.2 | 79.0 | 1.21x |
| p6 | Neural Engine | 141.5 | 97.7 (!) | 1.45x |
| fp16 | Neural Engine | 169.6 | 97.8 (!) | 1.73x |
| fp16 | ALL | 170.7 | 98.0 (!) | 1.74x |
| w8 | Neural Engine | 151.6 | 98.1 (!) | 1.55x |
| fastB | Neural Engine | 112.2 | 103.5 | 1.08x |
| fastA | Neural Engine | 125.1 | 115.3 | 1.08x |
| fp16 | GPU | 96.8 | 130.0 | 0.74x |
| p6 | GPU | 96.9 | 130.0 | 0.75x |
| fastA | GPU | 99.4 | 132.3 | 0.75x |
| PyTorch FP16 | MPS | 145.3 | 152.1 | 0.96x |
| fastB | CPU | 160.9 | 155.6 | 1.03x |
| fp16 | CPU | 133.4 | 155.9 | 0.86x |
| fp32 | GPU | 109.1 | 157.7 | 0.69x |
| fastB | GPU | 154.8 | 161.0 | 0.96x |
| fastC | CPU | 170.1 | 162.6 | 1.05x |
| fastC15_p6g | GPU | 176.8 | 168.2 | 1.05x |
| fastC_w8 | GPU | 176.2 | 168.6 | 1.05x |
| fastC_p6 | GPU | 176.8 | 168.6 | 1.05x |
| fastC15 | GPU | 176.8 | 168.7 | 1.05x |
| fastC | GPU | 176.8 | 169.1 | 1.05x |
| fastA | CPU | 183.7 | 192.6 | 0.95x |
| PyTorch FP32 | MPS | 137.9 | 205.5 | 0.67x |
| fp32 | CPU | 316.9 | 349.8 | 0.91x |
| PyTorch FP32 | CPU | 401.0 | 563.1 | 0.71x |
| w8 | GPU | 816.1 | 894.3 | 0.91x |
| PyTorch FP16 | CPU | 2669.3 | 3960.1 | 0.67x |
| fastC15_p4g | Neural Engine | 69.2 | not run | not run |
| p4 | GPU | 96.9 | not run | not run |
| a8w8 | GPU | 103.0 | not run | not run |
| w8 | CPU | 138.3 | not run | not run |
| p6 | CPU | 138.8 | not run | not run |
| p4 | Neural Engine | 141.2 | not run | not run |
| a8w8 | Neural Engine | 150.2 | not run | not run |
| a8w8 | CPU | 183.8 | not run | not run |
| p4 | CPU | 184.4 | not run | not run |
| fastC15_p4g | GPU | 187.3 | not run | not run |

## Sustained throughput

| Workers | M2 Pro images/s | M4 images/s | M4 speedup |
|---|---:|---:|---:|
| fp16_b4 on GPU (batch 4) | 11.01 | 7.51 | 0.68x |
| fp16 on GPU | 10.31 | 7.63 | 0.74x |
| fp16 on GPU + fp16 on GPU | 11.26 | 8.04 | 0.71x |
| fastC_b4 on NE (batch 4) | 10.11 | 11.99 | 1.19x |
| fastC on NE | 10.67 | 12.66 | 1.19x |
| fastC on NE + fastC on NE | 10.76 | 12.79 | 1.19x |
| fastC_w8 on NE | 13.36 | 16.43 | 1.23x |
| fastC15_p6g on NE | 14.54 | 18.0 | 1.24x |
| fastC on NE + fp16 on GPU | 21.0 | 20.03 | 0.95x |
| fastC on NE + fp16 on GPU + fp16 on CPU | 25.81 | 23.08 | 0.89x |
| fastC_w8 on NE + fp16 on GPU | 23.78 | 23.97 | 1.01x |
| fastC15_p6g on NE + fp16 on GPU | 24.98 | 25.43 | 1.02x |
| fastC_w8 on NE + fp16 on GPU + fp16 on CPU | 27.92 | 26.81 | 0.96x |

- **Throttling:** none on the M4. The end-of-run repeats gave GPU 7.63 to 7.58 images/s, and Neural Engine 12.66 to
  12.66 images/s.
- **Batching and extra copies:** these give little or nothing. A second copy on the same device adds 1% on the M4's
  Neural Engine and 5% on its GPU, and batch size 4 is slightly slower on both M4 devices.
- **Adding the CPU:** on the M4 this adds 12% to the 8-bit pair, but slows the Neural Engine worker from 16.5 to 14.2
  images/s.
- **Mixed workers:** different models produce slightly different embeddings, which matters if you compare embeddings
  across workers.

## Fidelity against PyTorch FP32 on CPU

The cosine columns show the mean and the worst image. The `Worst cosine` column in a run's `summary.md` throughput table
only checks the first image, so use this table instead.

| Model | Device | M2 Pro mean | M2 Pro worst | M2 Pro top-1 | M4 mean | M4 worst | M4 top-1 |
|---|---|---:|---:|---:|---:|---:|---:|
| a8w8 | GPU | 0.215403 | 0.129242 | 12% | not run | not run | not run |
| a8w8 | Neural Engine | -0.063777 | -0.142076 | 10% | not run | not run | not run |
| a8w8 | CPU | 0.211645 | 0.110587 | 8% | not run | not run | not run |
| fastA | GPU | 0.999991 | 0.999969 | 100% | 0.999991 | 0.999963 | 100% |
| fastA | Neural Engine | 0.999953 | 0.999791 | 100% | 0.999958 | 0.999845 | 100% |
| fastA | CPU | 0.999660 | 0.998628 | 100% | 0.999655 | 0.998681 | 100% |
| fastB | GPU | 0.999990 | 0.999956 | 100% | 0.999991 | 0.999957 | 100% |
| fastB | Neural Engine | 0.999951 | 0.999793 | 100% | 0.999957 | 0.999825 | 100% |
| fastB | CPU | 0.998661 | 0.997326 | 100% | 0.998661 | 0.997326 | 100% |
| fastC15 | GPU | 0.999990 | 0.999956 | 100% | 0.999990 | 0.999951 | 100% |
| fastC15 | Neural Engine | 0.999951 | 0.999793 | 100% | 0.999957 | 0.999825 | 100% |
| fastC15_p4g | GPU | 0.245115 | 0.153580 | 9% | not run | not run | not run |
| fastC15_p4g | Neural Engine | 0.808820 | 0.574834 | 100% | not run | not run | not run |
| fastC15_p6g | GPU | 0.996053 | 0.990188 | 100% | 0.996056 | 0.990120 | 100% |
| fastC15_p6g | Neural Engine | 0.995393 | 0.981301 | 100% | 0.995400 | 0.981666 | 100% |
| fastC | GPU | 0.999990 | 0.999956 | 100% | 0.999990 | 0.999951 | 100% |
| fastC | Neural Engine | 0.999951 | 0.999793 | 100% | 0.999957 | 0.999825 | 100% |
| fastC | CPU | 0.998629 | 0.996436 | 100% | 0.998629 | 0.996436 | 100% |
| fastC_p6 | GPU | 0.989831 | 0.975816 | 100% | 0.989829 | 0.975720 | 100% |
| fastC_p6 | Neural Engine | 0.989151 | 0.958877 | 100% | 0.989146 | 0.958734 | 100% |
| fastC_w8 | GPU | 0.998807 | 0.995855 | 100% | 0.998808 | 0.995874 | 100% |
| fastC_w8 | Neural Engine | 0.998791 | 0.996382 | 100% | 0.998791 | 0.996316 | 100% |
| fp16 | ALL | 0.999957 | 0.999780 | 100% | 0.999958 | 0.999845 | 100% |
| fp16 | GPU | 0.999991 | 0.999969 | 100% | 0.999991 | 0.999963 | 100% |
| fp16 | Neural Engine | 0.999953 | 0.999779 | 100% | 0.999958 | 0.999845 | 100% |
| fp16 | CPU | 0.999660 | 0.998628 | 100% | 0.999655 | 0.998681 | 100% |
| fp32 | GPU | 1.000000 | 1.000000 | 100% | 1.000000 | 1.000000 | 100% |
| fp32 | CPU | 1.000000 | 1.000000 | 100% | 1.000000 | 1.000000 | 100% |
| p4 | GPU | 0.326283 | 0.162502 | 33% | not run | not run | not run |
| p4 | Neural Engine | 0.325936 | 0.163464 | 32% | not run | not run | not run |
| p4 | CPU | 0.326533 | 0.163632 | 32% | not run | not run | not run |
| p6 | GPU | 0.989824 | 0.975883 | 100% | 0.989828 | 0.975921 | 100% |
| p6 | Neural Engine | 0.989154 | 0.959081 | 100% | 0.989811 | 0.976151 | 100% |
| p6 | CPU | 0.988517 | 0.957149 | 100% | not run | not run | not run |
| w8 | GPU | 0.998807 | 0.995971 | 100% | 0.998809 | 0.995978 | 100% |
| w8 | Neural Engine | 0.998794 | 0.996353 | 100% | 0.998790 | 0.996311 | 100% |
| w8 | CPU | 0.998396 | 0.991862 | 100% | not run | not run | not run |
| PyTorch FP16 | CPU | 0.999989 | 0.999941 | 100% | 0.999989 | 0.999942 | 100% |
| PyTorch FP16 | MPS | 0.999989 | 0.999953 | 100% | 0.999989 | 0.999949 | 100% |
| PyTorch FP32 | MPS | 1.000000 | 1.000000 | 100% | 1.000000 | 1.000000 | 100% |

- **8-bit weights** are the best trade-off: worst cosine is about 0.996.
- **6-bit grouped palettes** beat plain 6-bit palettes: worst cosine is about 0.98 against 0.96, at the same speed.
- **4-bit palettes** are not usable. Mean cosine is 0.33 on the original graph. The grouped version is 0.81 on the
  Neural Engine and 0.25 on the GPU.
- **Activation quantisation** (`a8w8`) scores at chance level. The cause wasn't diagnosed, so this doesn't show that
  activation quantisation can't work on this model.

## Rewrites for the Neural Engine

Rewrites keep the original weights, and are defined in [`pe_fast.py`](pe_fast.py). Medians on the Neural Engine:

| Graph | M2 Pro ms | M4 ms |
|---|---:|---:|
| Original FP16 | 169.6 | 97.8 (!) |
| Mode A: cleaned-up attention and RoPE | 125.1 | 115.3 |
| Mode B: channels-first, 1x1 convs | 112.2 | 103.5 |
| Mode C: B, with attention run per head | 93.2 | 79.0 |

The rewrites help much less on the M4. Mode C is 1.82x faster than the original graph on the M2 Pro and 1.24x on the M4,
where modes A and B are slower than the original. Modes B and C are also much slower than the original graph on the GPU,
so use each graph only on the device it suits.

## Neural Engine compile failures on the M4

On the M4, the original graph logs `ANECCompile() FAILED` on the Neural Engine, with FP16, 8-bit and 6-bit weights. Modes
A, B and C compile cleanly. The M2 Pro logged no such errors. CoreML doesn't report this as a failed prediction, so
`run_all.py` looks for the message in each step's output.

The flagged configs very likely still run on the Neural Engine, though this isn't proven:
- At about 98 ms they're faster than the CPU (155.9 ms), and `CPU_AND_NE` can't use the GPU.
- FP16's mean cosine of 0.999958 matches the rewritten models on the Neural Engine, not the GPU (0.999991) or CPU (0.999655).

Two things remain unexplained. All three weight types take the same 98 ms, although quantisation speeds up mode C. The
6-bit model's cosine (0.98981) is also closer to the GPU value than to the Neural Engine value for 6-bit weights (about
0.9892). The cause could be the chip or the macOS version, and these results can't separate them.

## Op placement

These results come from `MLComputePlan`, so they show CoreML's plan and not a measurement. The raw files are in
`reference/*/placement_*.txt`.

- Plain models are planned entirely on the chosen device, except that the failed compile on the M4 reports zero cost for
  every op, so the plan can't show what the failure changed.
- In the 6-bit grouped model, one conv and one cast are planned on the CPU, and that conv only supports the CPU. They are
  about 0.4% of estimated cost on the M2 Pro and 1.1% on the M4. This model is as fast as the plain 6-bit one, so it
  doesn't appear to cost anything.
- Cost shares for quantised models add up to only about 85-89%. The remainder wasn't traced.
- **Reference file caveat:** the M4 placement files come from before a counting fix in `profile_plan.py`. Their device
  switch counts of 204 and 209 are an artefact of counting weight-unpacking ops as a device. With the fix, the M2 Pro's
  equivalent models show 1 and 0.

## Hardware profile of the Neural Engine, M2 Pro only

These counters come from `profile_hw.py` and `micro_ane.py`. **None of this has been measured on an M4 yet**, and the M4's
limits may differ.

### Memory traffic per image

| Model | Device | Median ms | Reads per image | Writes per image | DRAM reads per image | Port GB/s | DRAM GB/s | Time above 30 GB/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fastC | Neural Engine | 94.1 | 3.41 GB | 0.56 GB | 1.47 GB | 42.3 | 15.9 | 70% |
| fastC_w8 | Neural Engine | 75.4 | 1.60 GB | 0.57 GB | 0.40 GB | 28.8 | 5.7 | 41% |
| fastC15_p6g | Neural Engine | 68.9 | 1.48 GB | 0.56 GB | 0.38 GB | 29.5 | 6.0 | 41% |
| fp16 | Neural Engine | 168.9 | 6.96 GB | 2.72 GB | 3.22 GB | 57.4 | 26.6 | 100% |
| fp16 | GPU | 96.4 | n/a | n/a | n/a | n/a | 10.2 (GPU) | n/a |

"Port" is all traffic through the Neural Engine's memory port, including system-cache hits. "DRAM" is the part that
reached memory. FP16 on the GPU moves far less data than FP16 on the Neural Engine.

### Memory contention

CPU threads copied 512 MB buffers (`dram`) or 256 KB buffers that stay in cache (`cache`), while the model ran on the
Neural Engine.

| Model | CPU threads | Load type | Median ms | Port GB/s | Total DRAM GB/s | CPU DRAM GB/s |
|---|---:|---|---:|---:|---:|---:|
| fastC | 0 | none | 93.9 | 42.3 | 19.5 | 2.1 |
| fastC | 2 | dram | 94.3 | 42.1 | 40.4 | 23.3 |
| fastC | 4 | dram | 94.4 | 41.9 | 63.4 | 40.5 |
| fastC | 6 | dram | 95.8 | 40.8 | 60.5 | 41.8 |
| fastC | 2 | cache | 93.9 | 42.1 | 18.1 | 1.4 |
| fastC | 4 | cache | 93.8 | 42.1 | 17.8 | 1.1 |
| fastC | 6 | cache | 93.9 | 41.9 | 17.7 | 1.2 |
| fastC | 0 | none | 93.6 | 42.4 | 18.4 | 1.6 |
| fp16 | 0 | none | 168.6 | 57.4 | 28.0 | 0.9 |
| fp16 | 2 | dram | 168.0 | 57.6 | 41.2 | 13.5 |
| fp16 | 4 | dram | 167.9 | 57.3 | 54.2 | 24.1 |
| fp16 | 6 | dram | 169.0 | 56.8 | 58.7 | 28.7 |
| fp16 | 2 | cache | 168.2 | 57.5 | 29.1 | 1.7 |
| fp16 | 4 | cache | 168.3 | 57.2 | 28.7 | 1.4 |
| fp16 | 6 | cache | 168.5 | 56.9 | 29.4 | 2.3 |
| fp16 | 0 | none | 168.5 | 57.4 | 28.5 | 1.2 |

Total DRAM traffic rose to about 60 GB/s with almost no change in latency, so shared DRAM isn't the limit at these
levels. That's about a third of the M2 Pro's rated 200 GB/s, so this doesn't test a fully saturated system.

### Peak throughput on synthetic models

Stacks of 1x1 convs with random weights, using shape `(1, C, 1, S)`. Median times of a few milliseconds include call
overhead, so these rates understate the peak, especially on the shortest runs.

| Channels | Tokens | Layers | Activations | Weights | Median ms | Trillion MAC/s | Port GB/s | DRAM GB/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 16 | 96 | 32 KB | 201.3 MB | 3.5 | 0.46 | 58.0 | 57.8 |
| 1024 | 64 | 48 | 128 KB | 100.7 MB | 1.9 | 1.72 | 53.7 | 53.5 |
| 1024 | 144 | 24 | 288 KB | 50.3 MB | 1.1 | 3.38 | 47.8 | 47.2 |
| 1024 | 288 | 16 | 576 KB | 33.6 MB | 1.2 | 3.96 | 28.4 | 27.8 |
| 1024 | 577 | 8 | 1154 KB | 16.8 MB | 1.2 | 3.88 | 36.6 | 14.9 |
| 512 | 577 | 24 | 577 KB | 12.6 MB | 0.8 | 4.53 | 17.6 | 16.6 |
| 256 | 577 | 64 | 288 KB | 8.4 MB | 0.6 | 3.89 | 15.0 | 14.1 |
| 256 | 2304 | 24 | 1152 KB | 3.1 MB | 1.3 | 2.75 | 37.4 | 4.2 |
| 128 | 4608 | 32 | 1152 KB | 1.0 MB | 1.5 | 1.62 | 41.7 | 2.4 |
| 64 | 9216 | 64 | 1152 KB | 0.5 MB | 2.4 | 1.02 | 51.4 | 1.4 |

These are plain conv stacks with call overhead included, so 4.53 is not this machine's peak. A split MLP reached 5.65 at
6.9 ms (`split_mlp.jsonl`). With overhead removed, [`aneopt probe slope`](https://github.com/ganessh22/aneopt/blob/main/OPTIMISATION.md#the-old-peak-of-45-t-macs-was-too-low) gives **8.45 trillion MAC/s** on a 7x7,
64-channel conv stack and 7.09 on a 1x1, 512-channel stack. 8.45 is the best of the graphs tried, not a limit of the chip.

### What this shows

- **The port appears capped near 60 GB/s.** The weight-heavy synthetic model sustains 58 GB/s and original FP16 sustains
  57 GB/s. FP16 on the Neural Engine looks essentially limited by this port.
- **Weights seem to be read about five times per image.** The model has about 604 MB of weights, but mode C reads 3.41
  GB, and activations can explain only about 0.56 GB of that if reads match writes. Tiling that re-reads weights for each
  tile would explain this, but that's a hypothesis. Quantising reduces this traffic sharply.
- **Time splits about evenly.** Across the three mode C models, latency fits roughly 47 ms + 12 ms per GB
  moved, using reads plus writes. Three points is a rough fit, but it implies about half of mode C's time is data movement.
- **Computing runs well below peak.** The model needs about 190.6 billion multiply-adds, counting patch-level linear,
  attention and MLP layers. That's about 2.0 trillion per second for mode C, or about 24% of the 8.45 trillion peak. At
  that rate the same work takes about 23 ms, or about 27 ms at the 7.09 trillion of a 1x1 stack, against the fit's 47 ms of
  non-transfer time. A real model has attention and other work that may never reach these rates, so treat them as a floor,
  not a target.
- **No counter gives "% of Neural Engine used".** The cluster stays powered about 99.9% of the time during inference, and
  a second copy of the model adds almost nothing, so the engine looks occupied rather than idle.

### Next things to try

- **Avoid chunking across tokens.** Weight reads look higher at 577 tokens than at 144 or fewer, but these counters round
  heavily at that scale, so this is only weak evidence.
- **Run the profilers and mode D on an M4.** Its port limit and on-chip limit are the key unknowns.

## Mode D, M2 Pro only

Measured after the tables above, on the same M2 Pro, with the rewrites run in turn in one session. The method and the general
lessons are in [`OPTIMISATION.md`](../../aneopt/OPTIMISATION.md), and the raw data is in `reference/m2pro/rewrites.jsonl`,
`working_set_sweep.jsonl` and `split_mlp.jsonl`.

| Model | Median ms | Reads per image | Writes per image | DRAM reads per image | Mean cosine | Worst cosine |
|---|---:|---:|---:|---:|---:|---:|
| fastC | 94.2 | 3.41 GB | 0.56 GB | 1.47 GB | 0.999951 | 0.999793 |
| fastDQ: split QKV | 96.1 | 3.49 GB | 0.65 GB | 1.45 GB | 0.999951 | 0.999793 |
| fastDM: split MLP | 85.4 | 2.58 GB | 0.66 GB | 1.48 GB | 0.999952 | 0.999776 |
| fastCP: two-stage patch conv | 84.0 | 2.73 GB | 0.56 GB | 0.78 GB | 0.999951 | 0.999794 |
| fastDS: split MLP and QKV | 83.6 | 2.77 GB | 0.68 GB | 1.45 GB | 0.999952 | 0.999776 |
| fastD: all splits, two-stage patch conv | 73.5 | 2.08 GB | 0.69 GB | 0.76 GB | 0.999951 | 0.999790 |
| fastD_w8c: 8-bit conv weights only | 68.5 | 1.48 GB | 0.60 GB | 0.44 GB | 0.999294 | 0.998269 |
| fastD_w8: all 8-bit weights | 66.8 | 1.28 GB | 0.60 GB | 0.38 GB | 0.998858 | 0.996314 |

- **Compile and top-1:** all of these logged no compile failures, and all score 100% top-1.
- **Repeats:** fastD, fastDS and fastD_w8 were run twice, with medians within 0.2 ms. Three runs of fastC gave 93.7, 94.2 and 94.7 ms.
- **Load time:** the two-stage patch conv raises loading in a new process from about 30 s to about 53 s. The cause wasn't found.
- **Against the old best:** fastD_w8c matches fastC15_p6g (68.3 ms) with much better fidelity. Its worst cosine is 0.9983 against 0.981.
- **Other rewrites of mode D without the patch change:** turning off the conv-bias and batch-norm fusion passes gave 83.7 and 83.6 ms, and
  so made no difference.
- **Not measured:** mode D on the GPU or on the M4. The M4 gained less from earlier rewrites, so don't assume the same gains there.

### The on-chip working-set limit

`micro_ane.py sweep` on 256-channel stacks shows a step between 1792 and 2048 KiB. Throughput falls from 2.41 to 1.57 trillion
MAC/s, writes per run rise from 0.018 to 0.067 GB, and the port reaches about 62 GB/s. Mode C has activations of about 4.98 MB and
3.74 MB in every block, and `aneopt audit` finds none over the limit in mode D. The 1024-channel stacks show a smaller drop, from 3.86
to 3.13 trillion MAC/s, between 1154 and 2048 KiB.

This explains part of the unexplained reads above, but not all of them. Mode D still reads 2.08 GB per image from about 0.6 GB of
weights.

### Comparison with ANEForge

ANEForge 0.4.0 compiles to private `e5rt` APIs without CoreML. The port in `pe_aneforge.py` mirrors modes C and D, and always uses the
two-stage patch conv, because the alternatives failed: `space_to_depth` by 7 or 14 wouldn't compile, and a 14x14 conv hit a Python
shape check.

| Graph | ANEForge ms | CoreML ms | ANEForge worst cosine | CoreML worst cosine |
|---|---:|---:|---:|---:|
| Mode C with two-stage patch conv | 84.6 | 84.0 | 0.999788 | 0.999794 |
| Mode D, without its pooling split | 74.1 | 73.5 (with it) | 0.999802 | 0.999790 |
| Mode D, 8-bit conv weights | 69.6 | 68.5 | 0.998319 | 0.998269 |

ANEForge compiles the full graph in about 55 s. Its zero-copy path takes about 0.8 ms less than a call with copies. Fidelity and traffic
closely match CoreML on the same graph, so we see no speed advantage from the direct route on this model.

## Projection for the M6 (not measured)

No M6 was tested, so this is only an estimate. [The M6](https://www.macrumors.com/2026/08/25/apple-reveals-m6/) has dual
16-core Neural Engines with up to 2x peak compute, which Apple says frameworks can use automatically, and a
[12-core GPU with Neural Accelerators](https://support.apple.com/en-us/128108). It isn't confirmed whether one model spans
both engines, or whether CoreML uses the accelerators for this kind of model.

| 8-bit on Neural Engine + FP16 on GPU | M4 measured | M6 low estimate | M6 high estimate |
|---|---:|---:|---:|
| Neural Engine images/s | 16.4 | 19.7 (1.2x) | 26.2 (1.6x) |
| GPU images/s | 7.5 | 9.8 (1.3x) | 20.3 (2.7x) |
| Total images/s | 24.0 | about 29 (1.2x) | about 46 (1.9x) |

- **Low estimate:** the GPU gains from its extra cores plus a small allowance for newer hardware, and one model gets little
  from the second engine.
- **High estimate:** the second engine and extra bandwidth both help, and the GPU matches one early Geekbench AI result.
  [Other early results](https://www.macwelt.de/article/3239298/mac-mini-m6-benchmarks-geekbench.html) put the Neural Engine
  about 50% above the M4. These are uncontrolled and don't use this model.

Run this benchmark on an M6 before relying on these figures. The two most useful results are the single-model Neural
Engine speed, and whether two copies now scale.

## Other limitations

- Results cover one machine of each type, and only Imagenette for fidelity. Test on your own data before choosing a
  lossy model.
- The over-800 ms result for the original graph with 8-bit weights on the GPU wasn't diagnosed.
- Neural Engine power wasn't measured, because it needs `sudo powermetrics`.
