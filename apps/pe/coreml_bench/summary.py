"""Write results/summary.md, comparing this machine to the bundled M2 Pro (16-core GPU) reference numbers."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


info = json.loads((RESULTS / "machine.json").read_text())
ref = {}
for r in jsonl(HERE / "reference" / "m2pro" / "results.jsonl"):
    ref.setdefault(r["tag"], r)  # first measurement, as the later repeats only check for drift
ref_tp = {}
for r in jsonl(HERE / "reference" / "m2pro" / "throughput.jsonl"):
    ref_tp.setdefault(" + ".join(sorted(w["spec"] for w in r["workers"])), r)

errors_file = RESULTS / "ane_compile_errors.txt"
compile_errors = [l for l in errors_file.read_text().splitlines() if l.strip()] if errors_file.exists() else []
flagged_tags = {l.replace("bench: ", "").replace(" ", "_") for l in compile_errors if l.startswith("bench: ")}
flagged_groups = {" + ".join(sorted(l.split(": ", 1)[1].split(" + "))) for l in compile_errors if l.startswith("throughput: ")}

out = ["# PE-Core-L14-336 CoreML benchmark", ""]
out += [f"- **{k}:** {v}" for k, v in info.items()]
if info["quick"]:
    out += ["", "**Quick mode: these timings are a smoke test and are not comparable.**"]

if compile_errors:
    out += ["", "## Neural Engine compile errors", ""]
    out += ["These steps logged a failed Neural Engine compile. CoreML carries on regardless, so results marked"]
    out += ["(!) below completed but may not be running purely on the Neural Engine.", "", "```"]
    out += compile_errors + ["```"]

out += ["", "## Latency, batch size 1", ""]
out += ["| Config | Median ms | Images/s | First-load s | M2 Pro ms | Speedup vs M2 Pro |", "|---|---:|---:|---:|---:|---:|"]
rows = jsonl(HERE / "results.jsonl")
for r in sorted(rows, key=lambda r: r["median_ms"]):
    base = ref.get(r["tag"])
    if r["tag"] in flagged_tags:
        r["tag"] += " (!)"
    out.append(
        f"| {r['tag']} | {r['median_ms']} | {r['fps']} | {r['load_s']} | {base['median_ms'] if base else '–'} | "
        f"{base['median_ms'] / r['median_ms']:.2f}x |" if base else
        f"| {r['tag']} | {r['median_ms']} | {r['fps']} | {r['load_s']} | – | – |"
    )

out += ["", "## Sustained throughput", ""]
out += ["| Workers | Total images/s | M2 Pro | Speedup | Worst cosine |", "|---|---:|---:|---:|---:|"]
for r in jsonl(HERE / "throughput.jsonl"):
    key = " + ".join(sorted(w["spec"] for w in r["workers"]))
    base = ref_tp.get(key)
    cos = min(w["cos_min"] for w in r["workers"])
    b = base["total_images_per_s"] if base else None
    speedup = f"{r['total_images_per_s'] / b:.2f}x" if b else "–"
    mark = " (!)" if key in flagged_groups else ""
    out.append(f"| {key}{mark} | {r['total_images_per_s']} | {b or '–'} | {speedup} | {cos} |")

fidelity = RESULTS / "fidelity.txt"
if fidelity.exists():
    out += ["", "## Fidelity vs PyTorch FP32 CPU", "", "```"]
    out += [l for l in fidelity.read_text().splitlines() if l.strip()]
    out += ["```"]

placements = sorted(RESULTS.glob("placement_*.txt"))
if placements:
    out += ["", "## Op placement, as planned by CoreML", ""]
    out += ["This is CoreML's plan, not a measurement. A compile error above can still change what actually runs.", "", "```"]
    for path in placements:
        lines = path.read_text().splitlines()
        out += [l for l in lines if l.endswith("non-const ops") or l.startswith(("cost share", "device switches"))] + [""]
    out += ["```"]

failures = (RESULTS / "failures.txt").read_text().strip() if (RESULTS / "failures.txt").exists() else ""
if failures:
    out += ["", "## Failures", "", "```", failures, "```"]

(RESULTS / "summary.md").write_text("\n".join(out) + "\n")
print("wrote results/summary.md")
