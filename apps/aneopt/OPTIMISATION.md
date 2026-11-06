# Optimising models for the Apple Neural Engine

What we learned making PE-Core-L14-336 faster on the Neural Engine, written so the method can be reused on other models.
The commands come from the [`aneopt`](README.md) package.

## Summary

- **Results:** on an M2 Pro, PE-Core-L14 went from 169.6 ms to 73.5 ms with almost no fidelity loss, and to 68.5 ms with
  8-bit conv weights. All of it runs through public CoreML, using the original weights.
- **The biggest wins came from graph shape:** a few specific ops caused most of the avoidable memory traffic, so find those
  first.
- **Bypassing CoreML gained nothing here:** the same graph ran at 73.5 ms on CoreML and 74.1 ms through a direct route.
- **Measure each change on its own:** the largest late gain came from an op we had assumed was too small to matter.

These numbers come from one M2 Pro on macOS 27.0.1. None of the newest changes have been run on any other chip. The raw data
is in [`reference/m2pro`](../pe/coreml_bench/reference/m2pro), and the wider benchmark is in
[`RESULTS.md`](../pe/coreml_bench/RESULTS.md).

| Step | Median ms | Reads per image | DRAM reads per image | Worst cosine |
|---|---:|---:|---:|---:|
| Direct conversion, FP16 | 169.6 | 6.96 GB | 3.22 GB | 0.99978 |
| Channels-first, 1x1 convs, attention per head (mode C) | 94.2 | 3.41 GB | 1.47 GB | 0.99979 |
| C + two-stage patch conv (CP) | 84.0 | 2.73 GB | 0.78 GB | 0.99979 |
| C + wide layers split by channel (DS) | 83.6 | 2.77 GB | 1.45 GB | 0.99978 |
| **Both changes (mode D)** | **73.5** | 2.08 GB | 0.76 GB | 0.99979 |
| Mode D, 8-bit conv weights only | 68.5 | 1.48 GB | 0.44 GB | 0.99827 |
| Mode D, all 8-bit weights | 66.8 | 1.28 GB | 0.38 GB | 0.99631 |

Cosine is against PyTorch FP32 on CPU, over 100 Imagenette images.

## The method

Apply these in order. Rebuild and measure after each change, one change at a time.

1. **Get a baseline.** Convert to FP16 and record latency and traffic, repeating each measurement.

   ```bash
   aneopt profile model.mlpackage
   ```

2. **Check placement.** A model partly on the CPU has a different problem, and the rest of this guide won't fix it.

   ```bash
   aneopt plan model.mlpackage
   ```

3. **Find this chip's on-chip limit.** Look for the throughput step.

   ```bash
   aneopt probe sweep
   ```

4. **List what exceeds it.** Use the limit from step 3.

   ```bash
   aneopt audit model.mlpackage --limit-kib 1792
   ```

5. **Fix the layout and the oversized tensors.** See the rules below.
6. **Compress last, and check fidelity on real data.** Compression can break a model without any error.
7. **Compare with a reference model on every build.** Check the rewrite in PyTorch first, because it should match almost
   exactly. Then check the CoreML output.

## What we learned

Confidence labels: **measured** means we saw it directly in this work, **published** means it comes from a cited source and
we didn't re-measure it, and **unexplained** means we measured the effect but don't know the cause.

### Keep each activation under the on-chip limit (measured)

An activation above the limit is streamed through the Neural Engine's memory port. Our sweep of 1x1 conv stacks with 256
channels shows a sharp step on the M2 Pro:

| Activation | Throughput | Writes per run | Port traffic |
|---:|---:|---:|---:|
| 1792 KiB | 2.41 trillion MAC/s | 0.018 GB | 31.7 GB/s |
| 2048 KiB | 1.57 trillion MAC/s | 0.067 GB | 62.0 GB/s |

Above the step, the port stays near 60 GB/s and latency grows with size. The limit depends on the chip: about 2 MB on M1 and
4.72 MB on M5, according to [Bryngelson's paper](https://arxiv.org/abs/2606.22283). Always measure yours.

### Split wide layers by channel (measured)

A wide linear layer's output can usually be computed in parts. Using the same weights, an expansion layer becomes several
narrower convs, with the activation applied per branch, and the next layer sums the branches. Use `SplitMLP`, `SplitOut` and
`SplitIn`.

| Test | Before | After |
|---|---:|---:|
| Synthetic MLP with a 4x hidden layer | 9.2 ms | 6.9 ms, with 41% fewer reads |
| Real model, MLP and QKV split | 93.7 ms | 83.6 ms |

Split the layers the audit flags, and measure them separately. In PE-Core-L14 the MLP split gave nearly all of the FP16 gain.
Splitting QKV alone was about 2 ms slower, yet with 8-bit weights the combined split was about 6 ms faster than the MLP split
alone. Rules like this one can reverse once compression changes the traffic.

This applies to any layer whose channel output is computed independently per channel. Typical cases are transformer MLPs,
QKV projections and inverted-bottleneck blocks in CNNs.

### Check large-kernel convs, even cheap ones (measured, cause unexplained)

Replacing the 14x14 stride-14 patch conv with an exact two-stage version saved 10 ms and removed about 0.69 GB of DRAM reads
per image. That op is only about 0.2% of the model's multiply-adds, and it was not over the on-chip limit. The saved traffic is
close to one full pass over the model's weights, but we don't know why the conv caused it.

`factor_patch_conv(conv, first)` does this rewrite exactly. Stage one uses identity kernels to move pixels into channels, and
stage two holds the original weights. Try it on any ViT-style patch embedding, and measure the result, since other kernel sizes
may behave differently.

**The cost:** loading the model in a new process went from about 30 s to about 53 s on this machine. We didn't find out why.
Check that this is acceptable for your application.

### Use the Neural Engine's layout (published, and measured here)

These come from Apple's [transformer](https://machinelearning.apple.com/research/neural-engine-transformers) and
[vision transformer](https://machinelearning.apple.com/research/vision-transformers) guidance. Together they took PE-Core-L14 from
169.6 to 94.2 ms.

- Use `(B, C, 1, S)`, with 1x1 convs in place of linear layers.
- Run attention once per head, with a softmax per head.
- Avoid reshapes and transposes. Arrange attention so that it needs just one transpose.
- The last axis is aligned to 64 bytes, so a small last axis wastes memory. `aneopt audit` includes this padding in its sizes.

The gains depend on the chip. On an M4, modes A and B were slower than the original graph, though mode C was still faster.

### Compress only the weights that matter (measured)

`coremltools` quantises every large constant by default, which includes positional tables as well as weights. Quantising only
conv weights raised the worst cosine from 0.9963 to 0.9983, at a cost of 1.7 ms. The default per-channel setting isn't the
issue, since asking for it explicitly gave identical output.

On the M2 Pro, mode D with 8-bit conv weights equals the previous fastest usable config (68.3 ms) with much better fidelity
(worst cosine 0.9983 against 0.981). Use `quantize_weights(src, dst, conv_only=True)`.

Compression helps most when a model is limited by bandwidth, because it reduces weight traffic. Bits and methods can fail badly:

- 4-bit palettes gave a mean cosine of 0.33 on the original graph.
- Activation quantisation scored at chance level on the original graph. We didn't find out why.
- Top-1 on an easy dataset hides this kind of damage, so compare cosine similarity as well.

### Try fusion passes last (measured)

Turning off `coremltools`' conv-bias and batch-norm fusion passes changed nothing: 83.7 and 83.6 ms, against 83.6 ms with them.
ANEForge's spelled-out LayerNorm with `rsqrt` also matched its default one, so small op-level differences here didn't matter.

### Don't expect gains from skipping CoreML (measured on this model)

CoreML's call overhead is small: a tiny model's whole prediction takes about 0.6 ms, which is under 1% of this model's latency.
Fusing into one program doesn't help either, since `MLComputePlan` already shows zero device switches.

We ported PE-Core-L14 to [ANEForge](https://github.com/sbryngelson/ANEForge) to test this directly:

| Graph | CoreML | ANEForge |
|---|---:|---:|
| Mode C with two-stage patch conv | 84.0 ms | 84.6 ms |
| Mode D | 73.5 ms | 74.1 ms |
| Mode D, 8-bit conv weights | 68.5 ms | 69.6 ms |

ANEForge first looked 10 ms faster, at 74.1 ms against CoreML's 83.6 ms. But its port needed the two-stage patch conv, because a
14x14 conv wouldn't get past its checks, and CoreML's graph didn't have it yet. That was how we found the patch conv gain. The ANEForge
mode D also lacks the pooling split. **Before crediting a runtime with a speedup, reproduce its exact graph in the other runtime.**

Direct routes remain useful for research. They expose compile options and let you try a graph quickly in about a minute. But they
rely on undocumented APIs that can break with any OS update, and the paper and ANEForge's own docs both point to CoreML for shipped
software.

### Other things that didn't help

These come from the earlier benchmark on the same M2 Pro and on an M4:

- Batch size 4 is slightly slower than batch size 1.
- A second copy of the model on the Neural Engine adds about 1% throughput.
- CPU load that raises total DRAM traffic to about 60 GB/s doesn't change latency, so shared DRAM isn't the limit at those levels.

## Applying this to another model

| Model type | Look at first |
|---|---|
| Vision transformer | Patch conv, MLP expansion, QKV projection, then per-head attention |
| Text transformer or LLM | MLP expansion and QKV. Long sequences make every activation wider, so short ones may fit on chip |
| Long-sequence attention | Scores of `S x S` per head can pass the limit. The [paper](https://arxiv.org/abs/2606.22283) found the GPU about 2.1x faster at sequence length 512, so compare both |
| CNN | Wide or high-resolution feature maps, where splitting by channel or tile may help. Avoid large or strided kernels. We haven't tested these |
| Any model with short sequences | Compute may be limited by weight bandwidth, so compression may help most. Apple's guidance covers this case |

To work out activation sizes by hand, multiply channels, tokens and bytes per element: FP16 at 4096 x 577 x 2 is 4.7 MB.

Rewrites that change a graph's memory traffic can make it slower on the GPU, as modes B and C were. Build a separate graph for each
device. We haven't measured mode D on the GPU.

## Tools

### In this package

| Tool | Use it for |
|---|---|
| `aneopt audit` | A list of oversized activations, before running anything |
| `aneopt plan` | Device placement and CoreML's cost estimates, which are estimates, not measurements |
| `aneopt profile` | Measured latency and memory traffic |
| `aneopt probe sweep` | This chip's on-chip limit |
| `aneopt probe mlp` | A quick test of whether splitting helps on this chip |
| `aneopt.layers` | The rewrites described above |

### External

| Tool | What it is | Our experience |
|---|---|---|
| [coremltools](https://apple.github.io/coremltools/) | Apple's converter, MIL builder, quantisation and `MLComputePlan` | The route to use for anything you ship. Tracing PyTorch was enough for every rewrite here |
| IOReport | Private macOS counters for memory traffic per SoC block, with no sudo needed | The most useful diagnostic here: it's what revealed the traffic differences. Channel names can vary by chip and OS |
| [ANEForge](https://github.com/sbryngelson/ANEForge) | Python frontend to private `e5rt` APIs, skipping CoreML | Good for fast experiments. Worked on macOS 27 for this graph. `space_to_depth` by 7 or 14 failed to compile, and a 14x14 conv hit a Python shape check. Not faster than CoreML here |
| [Bryngelson's paper](https://arxiv.org/abs/2606.22283) | 303-page study of Neural Engine architecture and performance | Led us to the working-set limit, which our sweep then confirmed on the M2 Pro. It has the same author as ANEForge, and its measurements are mostly on M1 and M5 |
| [Apple's research posts](https://machinelearning.apple.com/research/neural-engine-transformers) | Published guidance on transformers | The basis for modes B and C |
| [maderix/ANE](https://github.com/maderix/ANE), [Orion](https://github.com/mechramc/Orion) | Direct private-API runtimes, aimed at training and small LLMs | Read only, not run. Useful notes on limits, some of which didn't match what we saw: Orion lists `gelu` and `concat` as invalid, and both worked through ANEForge's route |
| [ANEMLL](https://github.com/Anemll/Anemll) | Converter from Hugging Face to ANE-optimised CoreML, focused on LLMs | Read only, not run |

## What we still don't know

- **Other chips:** the limit, the split gain and the patch conv gain are unmeasured beyond the M2 Pro. Rerun the sweep and benchmark on
  each chip you care about. An M4 is the obvious next test.
- **The patch conv:** why this one op caused so much traffic, why the rewrite slows loading, and whether it depends on the kernel size.
- **The remaining traffic:** mode D still reads 2.08 GB per image from about 0.6 GB of weights. With 8-bit weights it reads 1.28 GB from about
  0.3 GB. Something other than oversized activations accounts for this.
- **Real data:** fidelity here is from 100 Imagenette images. Test lossy models on your own data.
- **Power:** we didn't measure it, because `powermetrics` needs sudo.
