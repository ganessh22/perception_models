"""CoreML's per-op device placement and estimated cost, from MLComputePlan. This is CoreML's plan, not a measurement."""
import shutil
from collections import defaultdict
from pathlib import Path

SHORT = {"MLNeuralEngineComputeDevice": "ANE", "MLGPUComputeDevice": "GPU", "MLCPUComputeDevice": "CPU"}


def print_plan(path, unit="CPU_AND_NE"):
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan

    unit_enum = getattr(ct.ComputeUnit, unit)
    compiled = ct.utils.compile_model(str(path))
    try:
        plan = MLComputePlan.load_from_path(compiled, compute_units=unit_enum)
        ops = plan.model_structure.program.functions["main"].block.operations

        stats = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
        device_cost = defaultdict(float)
        sequence = []
        for op in ops:
            if op.operator_name == "const":
                continue
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            cost = plan.get_estimated_cost_for_mlprogram_operation(op)
            device = SHORT.get(type(usage.preferred_compute_device).__name__, "?") if usage else "none"
            weight = cost.weight if cost else 0.0
            stats[op.operator_name][device][0] += 1
            stats[op.operator_name][device][1] += weight
            device_cost[device] += weight
            sequence.append((op.operator_name, device, usage))

        print(f"{Path(path).stem} / {unit}: {len(sequence)} non-const ops")
        print("cost share by device:", {k: round(v, 4) for k, v in device_cost.items()})
        print(f"\n{'op':32s} {'device':8s} {'count':>6s} {'cost':>8s}")
        rows = [(o, d, c, w) for o, ds in stats.items() for d, (c, w) in ds.items()]
        for o, d, c, w in sorted(rows, key=lambda r: -r[3]):
            print(f"{o:32s} {d:8s} {c:6d} {w:8.4f}")

        # Each device switch costs a tensor copy and usually a surface conversion, so count them.
        # Weight-unpacking ops report no device, so leave them out or each would count as two switches.
        placed = [s for s in sequence if s[1] != "none"]
        switches = sum(1 for a, b in zip(placed, placed[1:]) if a[1] != b[1])
        print(f"\ndevice switches along the op sequence: {switches}")
        print("non-ANE ops, and whether the ANE was supported for them:")
        seen = defaultdict(int)
        for name, device, usage in sequence:
            if device != "ANE" and usage:
                supported = sorted(SHORT.get(type(d).__name__, "?") for d in usage.supported_compute_devices)
                seen[(name, device, tuple(supported))] += 1
        for (name, device, supported), count in sorted(seen.items(), key=lambda kv: -kv[1]):
            print(f"  {name:28s} on {device:4s} x{count:<4d} supported={list(supported)}")
    finally:
        shutil.rmtree(compiled, ignore_errors=True)
