#!/usr/bin/env python
"""Aggregate latency reruns: mean ± std (sample std, n = #reps) per model/dataset, the v8s vs 11s gap,
and a comparison with the original (busy-GPU) run in results/.

Usage: python summarize_reruns.py reruns/<RUN_ID>
Writes: <RUN_ID>/latency_summary.md, <RUN_ID>/latency_errorbars.png
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(sys.argv[1]).resolve()
PROJ = Path(__file__).resolve().parents[1]
MODELS, DATASETS = ("yolov8s", "yolo11s", "rtdetr-l"), ("coco", "kitti")
KEYS = ("preprocess", "inference", "postprocess", "total")

reps = defaultdict(list)  # (model, ds) -> list of metrics dicts, one per rep
for f in sorted(OUT.glob("rep*/*/metrics.json")):
    m = json.loads(f.read_text())
    reps[(Path(m["config"]["model"]).stem, m["config"]["dataset"])].append(m)
assert reps, f"no rep*/*/metrics.json under {OUT}"
orig = {}
for f in (PROJ / "results").glob("*/metrics.json"):
    m = json.loads(f.read_text())
    orig[(Path(m["config"]["model"]).stem, m["config"]["dataset"])] = m


def ms(vals):
    a = np.asarray(vals, float)
    return a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0), len(a)


def fmt(vals, d=2):
    m, s, _ = ms(vals)
    return f"{m:.{d}f} ± {s:.{d}f}"


n_reps = max(len(v) for v in reps.values())
e = next(iter(reps.values()))[0]["env"]
L = ["# Latency reruns on an idle GPU", "",
     f"{n_reps} repetitions of every model × dataset (interleaved by rep), batch 1, FP32, imgsz 640, "
     "conf 0.001, NMS IoU 0.7 (YOLO only), max_det 300, 20 warm-up passes. "
     "Each repetition times every image in the dataset; values are **mean ± sample std across repetitions** (ms/img).", "",
     f"Hardware: {e.get('gpu')}, driver {e.get('driver')}, {e.get('os')}, torch {e.get('torch')}, "
     f"ultralytics {e.get('ultralytics')}. Run id `{e.get('run_id')}`.", ""]

gs = OUT / "gpu_state.csv"
if gs.exists():
    rows = list(csv.DictReader(gs.open()))
    u = [float(r["util_before_pct"]) for r in rows]
    mm = [float(r["mem_before_mib"]) for r in rows]
    L += [f"GPU state sampled before each of the {len(rows)} evaluations (5 × 1 s mean): "
          f"utilization {min(u):.0f}–{max(u):.0f}% (mean {np.mean(u):.1f}%), "
          f"memory used {min(mm):.0f}–{max(mm):.0f} MiB.", ""]

L += ["## 1. Latency, mean ± std over repetitions (ms/img)", "",
      "| Model | Dataset | n | Preprocess | Inference | Postprocess | Total | FPS (end-to-end) |",
      "|---|---|---|---|---|---|---|---|"]
for mod in MODELS:
    for ds in DATASETS:
        rs = reps.get((mod, ds))
        if not rs:
            continue
        cells = [fmt([r["speed_ms"][k] for r in rs]) for k in KEYS]
        fps = fmt([r["speed_ms"]["fps_end_to_end"] for r in rs], 1)
        L.append(f"| {mod} | {ds} | {len(rs)} | " + " | ".join(cells) + f" | {fps} |")

L += ["", "## 2. Is the YOLOv8s vs YOLO11s gap real?", "",
      "Δ = YOLO11s − YOLOv8s total latency. SE = √(s₁²/n₁ + s₂²/n₂). "
      "|Δ| > 2·SE is treated as a real difference (≈95%); otherwise it is within run-to-run noise.", "",
      "| Dataset | YOLOv8s total | YOLO11s total | Δ (ms) | 2·SE (ms) | Verdict |", "|---|---|---|---|---|---|"]
for ds in DATASETS:
    a, b = reps.get(("yolov8s", ds)), reps.get(("yolo11s", ds))
    if not (a and b):
        continue
    (m1, s1, n1), (m2, s2, n2) = ms([r["speed_ms"]["total"] for r in a]), ms([r["speed_ms"]["total"] for r in b])
    d, se = m2 - m1, np.sqrt(s1**2 / n1 + s2**2 / n2)
    verdict = "real (YOLO11s slower)" if d > 2 * se else "real (YOLO11s faster)" if d < -2 * se else "within noise"
    if min(n1, n2) < 2:
        verdict = "need ≥2 reps"
    L.append(f"| {ds} | {m1:.2f} ± {s1:.2f} | {m2:.2f} ± {s2:.2f} | {d:+.2f} | {2 * se:.2f} | {verdict} |")

L += ["", "## 3. Original run (busy GPU, ~35% util / ~2 GB from other apps) vs idle reruns", "",
      "| Model | Dataset | Original total | Rerun total (mean ± std) | Change |", "|---|---|---|---|---|"]
for mod in MODELS:
    for ds in DATASETS:
        rs, o = reps.get((mod, ds)), orig.get((mod, ds))
        if not (rs and o):
            continue
        m, s, _ = ms([r["speed_ms"]["total"] for r in rs])
        ot = o["speed_ms"]["total"]
        L.append(f"| {mod} | {ds} | {ot:.2f} | {m:.2f} ± {s:.2f} | {m - ot:+.2f} ({100 * (m - ot) / ot:+.0f}%) |")

L += ["", "## 4. Accuracy sanity check (should be identical across reps and to the original run)", "",
      "| Model | Dataset | Shared-class AP per rep | Original |", "|---|---|---|---|"]
for mod in MODELS:
    for ds in DATASETS:
        rs, o = reps.get((mod, ds)), orig.get((mod, ds))
        if not rs:
            continue
        aps = ", ".join(f"{100 * r['shared_classes']['stats']['AP']:.2f}" for r in rs if "shared_classes" in r)
        oap = f"{100 * o['shared_classes']['stats']['AP']:.2f}" if o and "shared_classes" in o else "n/a"
        L.append(f"| {mod} | {ds} | {aps} | {oap} |")

L += ["", "## 5. Commands", ""]
for (mod, ds), rs in sorted(reps.items()):
    L.append(f"- **{mod} / {ds}**: `{rs[0]['config']['command']}` (× {len(rs)}, `--out` differs per rep) | "
             f"weights md5 `{rs[0]['config']['weights_md5']}`")

(OUT / "latency_summary.md").write_text("\n".join(L) + "\n")

fig, ax = plt.subplots(figsize=(6.5, 4))
for ds, mk in (("coco", "o"), ("kitti", "s")):
    for mod in MODELS:
        rs = reps.get((mod, ds))
        if not rs or "shared_classes" not in rs[0]:
            continue
        m, s, _ = ms([r["speed_ms"]["total"] for r in rs])
        ap = 100 * rs[0]["shared_classes"]["stats"]["AP"]
        ax.errorbar(m, ap, xerr=s, fmt=mk, capsize=4, color={"coco": "C0", "kitti": "C1"}[ds],
                    label=f"{ds} (person/car/truck)" if mod == MODELS[0] else None)
        ax.annotate(mod, (m, ap), textcoords="offset points", xytext=(6, 4), fontsize=8)
ax.set_xlabel(f"End-to-end latency, ms/img (batch 1), mean ± std over {n_reps} reps")
ax.set_ylabel("AP@[.5:.95] (%)")
ax.set_title("Speed–accuracy, idle-GPU reruns"); ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "latency_errorbars.png", dpi=200)
print((OUT / "latency_summary.md").read_text())
