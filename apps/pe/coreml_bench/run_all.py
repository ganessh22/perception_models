"""Build every model, run every benchmark one at a time, then write results/summary.md and a results zip.

A config that fails or times out is logged and skipped, so one bad ANE compile can't lose the whole run.
Models that already exist are reused, so re-running after an interruption is cheap.
"""
import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PY = sys.executable
F, C = "pe_l14_fp16", "pe_l14_fastC"
W, G = "pe_l14_fastC_w8", "pe_l14_fastC15_p6g"
FAILURES = []
# A failed ANE compile doesn't fail the prediction: CoreML carries on, so it has to be spotted in the output.
ANE_ERROR_MARKERS = ("ANECCompile() FAILED", "MILCompilerForANE error")
COMPILE_ERRORS = []


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def machine_info():
    gpu = [l.strip() for l in sh(["system_profiler", "SPDisplaysDataType"]).splitlines() if "Total Number of Cores" in l]
    return dict(
        chip=sh(["sysctl", "-n", "machdep.cpu.brand_string"]),
        model=sh(["sysctl", "-n", "hw.model"]),
        memory_gb=int(sh(["sysctl", "-n", "hw.memsize"]) or 0) // 2**30,
        gpu_cores=gpu[0].split(":")[-1].strip() if gpu else "unknown",
        macos=platform.mac_ver()[0],
        low_power_mode="lowpowermode 1" in " ".join(sh(["pmset", "-g"]).split()),
        power_source=(sh(["pmset", "-g", "batt"]).splitlines() or ["unknown"])[0],
    )


def step(label, cmd, timeout, log=None):
    print(f"[{time.strftime('%H:%M:%S')}] {label}", flush=True)
    try:
        p = subprocess.run([PY, *cmd], cwd=HERE, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        FAILURES.append(f"{label}: timed out after {timeout}s")
        print("    TIMED OUT", flush=True)
        return False
    with open(RESULTS / "log.txt", "a") as f:
        f.write(f"\n=== {label}\n{p.stdout}\n{p.stderr}\n")
    if any(m in p.stdout + p.stderr for m in ANE_ERROR_MARKERS):
        COMPILE_ERRORS.append(label)
        print("    WARNING: Neural Engine compile error, so this result may not be pure Neural Engine", flush=True)
    if p.returncode != 0:
        FAILURES.append(f"{label}: exit code {p.returncode}")
        print(f"    FAILED (see results/log.txt): {p.stderr.strip().splitlines()[-1] if p.stderr.strip() else ''}", flush=True)
        return False
    for line in p.stdout.splitlines():
        if line.startswith("{"):
            print("    " + line, flush=True)
    if log:
        (RESULTS / log).write_text(p.stdout)
    return True


def build(script, last_artifact, timeout=3600, args=()):
    if (HERE / "models" / f"{last_artifact}.mlpackage").exists():
        print(f"reusing models from {script}", flush=True)
        return
    step(f"build: {script} {' '.join(args)}".strip(), [script, *args], timeout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    previous = HERE / "results_previous"
    if RESULTS.exists():
        shutil.rmtree(previous, ignore_errors=True)
        RESULTS.rename(previous)
    RESULTS.mkdir()
    for stale in ("results.jsonl", "throughput.jsonl"):
        (HERE / stale).unlink(missing_ok=True)
    shutil.rmtree(HERE / "embeds", ignore_errors=True)

    info = machine_info()
    info["quick"] = args.quick
    (RESULTS / "machine.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2), flush=True)
    if info["low_power_mode"]:
        print("WARNING: Low Power Mode is on, which will reduce these numbers.", flush=True)
    if "AC Power" not in info["power_source"]:
        print("WARNING: not on AC power, which can reduce these numbers.", flush=True)

    start = time.time()
    build("export_coreml.py", "pe_l14_fp32")
    build("pe_fast.py", "pe_l14_fastC")
    if not args.quick:
        build("quantize.py", "pe_l14_p6", args=("w8", "p6"))
        build("variants.py", "pe_l14_fastC15_p6g")

    timing = ["--n", "5", "--warmup", "1", "--eval", "5"] if args.quick else ["--eval", "100"]
    bench = lambda *cfg: step("bench: " + " ".join(cfg), ["bench.py", *cfg, *timing], 1800)

    # Always the full set: fidelity.py and throughput.py both index into this reference.
    step("bench: torch fp32 cpu (reference)", ["bench.py", "torch", "fp32", "cpu", "--eval", "500"], 3600)
    bench("torch", "fp32", "mps")
    bench("torch", "fp16", "mps")
    for unit in ("CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"):
        bench("coreml", F, unit)
        bench("coreml", C, unit)
    if not args.quick:
        bench("torch", "fp16", "cpu")
        bench("coreml", F, "ALL")
        for stem in ("pe_l14_fp32",):
            for unit in ("CPU_ONLY", "CPU_AND_GPU"):
                bench("coreml", stem, unit)
        for stem in ("pe_l14_fastA", "pe_l14_fastB"):
            for unit in ("CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"):
                bench("coreml", stem, unit)
        for stem in ("pe_l14_w8", "pe_l14_p6", "pe_l14_fastC_w8", "pe_l14_fastC_p6", "pe_l14_fastC15",
                     "pe_l14_fastC15_p6g"):
            for unit in ("CPU_AND_GPU", "CPU_AND_NE"):
                bench("coreml", stem, unit)

    secs = "5" if args.quick else "30"
    groups = [[f"{F}:CPU_AND_GPU:1"], [f"{C}:CPU_AND_NE:1"], [f"{C}:CPU_AND_NE:1", f"{F}:CPU_AND_GPU:1"]]
    if not args.quick:
        groups += [
            [f"{F}_b4:CPU_AND_GPU:4"],
            [f"{C}_b4:CPU_AND_NE:4"],
            [f"{C}:CPU_AND_NE:1", f"{C}:CPU_AND_NE:1"],
            [f"{F}:CPU_AND_GPU:1", f"{F}:CPU_AND_GPU:1"],
            [f"{C}:CPU_AND_NE:1", f"{F}:CPU_AND_GPU:1", f"{F}:CPU_ONLY:1"],
            [f"{W}:CPU_AND_NE:1"],
            [f"{G}:CPU_AND_NE:1"],
            [f"{W}:CPU_AND_NE:1", f"{F}:CPU_AND_GPU:1"],
            [f"{G}:CPU_AND_NE:1", f"{F}:CPU_AND_GPU:1"],
            [f"{W}:CPU_AND_NE:1", f"{F}:CPU_AND_GPU:1", f"{F}:CPU_ONLY:1"],
        ]
    for group in groups:
        step("throughput: " + " + ".join(group), ["throughput.py", "--secs", secs, *group], 1200)

    if not args.quick:
        # Same configs as earlier: a drop here means the machine throttled during the run.
        step("drift check: GPU", ["throughput.py", "--secs", secs, f"{F}:CPU_AND_GPU:1"], 1200)
        step("drift check: ANE", ["throughput.py", "--secs", secs, f"{C}:CPU_AND_NE:1"], 1200)

    # After the timed runs, so compiling these can't affect any timing.
    placements = [F, C] if args.quick else [F, C, "pe_l14_fastC_w8", G]
    for stem in placements:
        step(f"placement: {stem} CPU_AND_NE", ["profile_plan.py", stem, "CPU_AND_NE"], 1800, log=f"placement_{stem}_CPU_AND_NE.txt")
    step(f"placement: {F} CPU_AND_GPU", ["profile_plan.py", F, "CPU_AND_GPU"], 1800, log=f"placement_{F}_CPU_AND_GPU.txt")

    step("fidelity", ["fidelity.py"], 900, log="fidelity.txt")

    for name in ("results.jsonl", "throughput.jsonl"):
        if (HERE / name).exists():
            shutil.copy(HERE / name, RESULTS / name)
    (RESULTS / "failures.txt").write_text("\n".join(FAILURES) + "\n")
    (RESULTS / "ane_compile_errors.txt").write_text("\n".join(COMPILE_ERRORS) + "\n")
    step("summary", ["summary.py"], 60)

    chip = info["chip"].replace(" ", "_")
    archive = shutil.make_archive(str(HERE / f"pe_coreml_results_{chip}"), "zip", RESULTS)
    print(f"\nFinished in {(time.time() - start) / 60:.0f} min, {len(FAILURES)} failure(s), {len(COMPILE_ERRORS)} step(s) with Neural Engine compile errors.")
    print(f"Summary:  {RESULTS / 'summary.md'}")
    print(f"To share: {archive}")


if __name__ == "__main__":
    main()
