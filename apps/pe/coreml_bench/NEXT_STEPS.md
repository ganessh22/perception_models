# Mode E: where this stands and what's left

Paused on 2026-11-05. Nothing has been run beyond what's listed under "Where it stands".

## Where it stands

- Mode E is mode D with `aneopt.layers.attend_cf` in all 24 blocks. The pooling attention still uses `attend`.
- On the M2 Pro, E runs at 55.5 ms against D's 72.9 ms, with worst cosine 0.999795 against 0.999790. The full table is in
  [RESULTS.md](RESULTS.md), and raw data is in `reference/m2pro/attend_cf.jsonl`.

## Before picking this up again

- **Check `data` still works.** If it's a symlink to a folder that has gone, remove it and run `bash setup.sh`, which downloads Imagenette (about 340 MB).
- **Point `aneopt` at the right copy.** `setup.sh` installs the older `apps/aneopt`, which has no `attend_cf`, so `pe_fast.py E` would fail. After running it, reinstall the newer one:

  ```bash
  uv pip install --python .venv/bin/python --no-deps "git+https://github.com/ganessh22/aneopt.git"
  ```

- **Check for other Neural Engine jobs.** A DART-ane build was running during the A/B on 2026-11-05. For cleaner numbers, run when nothing else is using the machine:

  ```bash
  pgrep -lf 'ane/build.py|DART-ane'
  ```

- **Set `PYTHONPATH`.** Scripts that import `core` need the repo root. Run this from this folder:

  ```bash
  export PYTHONPATH="$(cd ../../.. && pwd)"
  ```

- **Disk:** about 16 GB was free when this was written. Each FP16 package is about 600 MB.

## Questions still open

1. **Why is E faster?** Reads fall 12%, DRAM reads don't change, MACs are identical, and E has more ops (3887 against 2855). Less traffic doesn't obviously account for 17 ms.
2. **Does compression still help on E?** Mode D gained about 5 ms from 8-bit conv weights, while RF-DETR gained almost nothing once compute-limited. `diagnose` now calls both D and E compute-limited, and it
   also called D compute-limited even though D gained from compression, so that verdict may be wrong for this model.
3. **Does attention in the pooling layer matter?** It still uses `attend`. It has one query, so a small gain at most.
4. **Does tiling help on E?** The 577 x 577 score maps fit on chip, and tiling a map that fits has been slower elsewhere, so this will probably not help.
5. **Is anything left to gain in attention?** `aneopt.attribute` hasn't been run on PE-Core. E is at about 41% of the 8.45 T MAC/s peak, so about 33 ms separates it from the floor of about 23 ms.
6. **Are the earlier mode C and D splits still worth having on E?** They were measured with `attend`. The QKV split in particular was slower alone, and rules like this can reverse.
7. **Does it hold on other chips and devices?** The M4 and the GPU are untested. Modes B and C were slower on the GPU, and the M4 gained less from earlier rewrites.
8. **Does E load faster or slower than D?** Load times were too noisy to compare, at 45-107 s for E and 56-533 s for D. They were taken with another build running.
9. **Why did the microbenchmark move?** `attend_cf` went from 0.538 to 0.411 ms, and `attend` batched went from 2.058 to 5.22 ms, between runs. That run shared the machine, so it may be contention.
10. **Is the `diagnose` advice right for this model?** The thresholds come from two models, and question 2 tests them.

## Things to run

In rough order of value. Each one needs a fresh D-against-E style A/B, in alternating new processes.

### 1. 8-bit weights on mode E

Answers question 2. If it holds, this is the new fastest usable config.

```python
from aneopt.export import quantize_weights

quantize_weights("models/pe_l14_fastE.mlpackage", "models/pe_l14_fastE_w8c.mlpackage", conv_only=True)
quantize_weights("models/pe_l14_fastE.mlpackage", "models/pe_l14_fastE_w8.mlpackage")
```

Compare against `pe_l14_fastD_w8c` (68.5 ms, worst cosine 0.998269) and `pe_l14_fastD_w8` (66.8 ms, 0.996314), which are already in `models/`. Score them with:

```bash
.venv/bin/python bench.py coreml pe_l14_fastE_w8c CPU_AND_NE --eval 100
```

```bash
.venv/bin/python fidelity.py
```

### 2. Rerun the A/B on a quiet machine

Confirms the 17.4 ms and gives a usable load-time comparison (question 8). Repeat this for each model, swapping the order each round:

```bash
.venv/bin/python profile_hw.py pe_l14_fastE CPU_AND_NE 15
```

Also rerun the microbenchmark (question 9):

```bash
.venv/bin/aneopt probe attention --heads 16 --dim 64 --tokens 577
```

### 3. Rerun `diagnose` with the new peak

`diagnose` here used 4.51 T MAC/s from earlier work. `aneopt probe slope` now gives 8.45 with call overhead removed, and the 41% for E is converted from that run, not measured again.
Only `share_of_peak` changes. This needs an `aneopt` from its `main` branch, not the older `apps/aneopt`.

```bash
.venv/bin/aneopt diagnose models/pe_l14_fastE.mlpackage --peak-tmac 8.45
```

### 4. Rerun on an M4

Answers question 7. This machine is M2 Pro only. Build E here and copy it across, or run `pe_fast.py E` there, then run the same A/B and `aneopt probe sweep`.

### 5. Mode E on the GPU

Also question 7, and this needs no other hardware:

```bash
.venv/bin/python bench.py coreml pe_l14_fastE CPU_AND_GPU --eval 100
```

Run the same for `pe_l14_fastD`, which has never been measured on the GPU either.

### 6. Remove each split from E, one at a time

Answers question 6. Add modes that are E without the QKV split, without the MLP split, and without the pooling split.

### 7. Attribute what's left

Answers question 5. Strip attention and the MLP out of E, then compare time and MAC rate using `aneopt.attribute`, as was done for RF-DETR.

### 8. Pooling attention

Answers question 3. It takes one line, but expect little: use `attend_cf` in `forward_b`'s pooling step.

## Housekeeping

- **Update `apps/aneopt`, or `setup.sh`.** Until one of them changes, a fresh `setup.sh` can't build mode E. Once either changes, `attend_cf` can become a normal import at the top of `pe_fast.py`.
- **Add mode E to `run.sh`.** `run_all.py` doesn't build or measure it.
- **Reword the `OPTIMISATION.md` summary if question 2 changes it.** It says compression has only been measured without `attend_cf`, and that PE-Core started out memory-limited and became compute-limited.
