# aneopt

Tools for finding and removing Apple Neural Engine bottlenecks in CoreML models. They grew out of optimising
PE-Core-L14, and they apply to other models, especially transformers.

The method, the findings and the tools it builds on are in [OPTIMISATION.md](https://github.com/ganessh22/aneopt/blob/main/OPTIMISATION.md).

This is an older copy of the package, which `setup.sh` installs. The current one, with `attend_cf` and `probe slope`, is in the [`aneopt` repository](https://github.com/ganessh22/aneopt).

## Install

Needs an Apple Silicon Mac and Python 3.10 or newer.

```bash
pip install ./apps/aneopt
```

From this directory, `pip install -e ".[test]"` gives an editable install with test dependencies, and `pytest tests` runs the tests.
None of the tests need the Neural Engine.

## Command line

```bash
aneopt audit model.mlpackage
```

```bash
aneopt plan model.mlpackage
```

```bash
aneopt profile model.mlpackage --seconds 12
```

```bash
aneopt probe sweep
```

| Command | What it tells you | Needs the Neural Engine |
|---|---|---|
| `audit` | Which activations exceed the on-chip limit, by op and shape. Change the limit with `--limit-kib` | No |
| `plan` | CoreML's per-op device placement and estimated cost | No |
| `profile` | Median latency and Neural Engine memory traffic per prediction, using random inputs | Yes |
| `probe sweep` | This machine's on-chip limit, found from a throughput step | Yes |
| `probe mlp` | Whether splitting a wide layer helps on this machine | Yes |
| `probe grid` | Peak throughput across activation and weight sizes | Yes |

`profile` and `probe` read IOReport, a private Apple API. It needs no sudo, but its channel names can differ across chips and macOS versions.

## Library

```python
from aneopt.layers import SplitMLP, conv1x1, attend_per_head, factor_patch_conv
from aneopt.export import convert, quantize_weights
from aneopt.audit import audit_package
from aneopt.measure import profile_package

# Replace a 14x14 stride-14 patch conv with an exact two-stage conv.
model.patch_embed = factor_patch_conv(model.patch_embed, 7)

# Replace an MLP with a 4x hidden layer. Input and output are channels-first, (B, C, 1, S).
mlp = SplitMLP(fc.weight.data, fc.bias.data, proj.weight.data, proj.bias.data)

convert(model, example, "model.mlpackage")
print(audit_package("model.mlpackage").over())
print(profile_package("model.mlpackage", inputs={"input": example.numpy()}))
quantize_weights("model.mlpackage", "model_w8.mlpackage", conv_only=True)
```

| Module | Contents |
|---|---|
| `aneopt.layers` | `conv1x1`, `LayerNormCF`, `SplitOut`, `SplitIn`, `SplitMLP`, `RotateHalf`, `factor_patch_conv`, `attend`, `attend_per_head` |
| `aneopt.audit` | `audit_package`, `print_audit`, `DEFAULT_LIMIT_BYTES` |
| `aneopt.export` | `convert`, `quantize_weights`, `palettize_weights` |
| `aneopt.measure` | `profile_package`, `measure`, `hist_stats` |
| `aneopt.plan` | `print_plan` |
| `aneopt.probe` | `run`, `find_step` |
| `aneopt.ioreport` | `Sampler`, with no dependencies beyond macOS |

The layers don't rewrite a model automatically. You still restructure its forward pass, as
[`pe_fast.py`](../pe/coreml_bench/pe_fast.py) does for PE-Core-L14.
