#!/usr/bin/env python
"""Collect results/*/metrics.json into results/summary.md (+ speed_accuracy.png) for the report."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parents[1] / "results"
runs = {}
for f in sorted(RES.glob("*/metrics.json")):
    m = json.loads(f.read_text())
    runs[(Path(m["config"]["model"]).stem, m["config"]["dataset"])] = m
models = sorted({k[0] for k in runs})


def pct(x):
    return "n/a" if x is None or x < 0 else f"{100 * x:.1f}"


L = ["# Results summary", "", "All runs: COCO-pretrained weights, no fine-tuning, imgsz 640, conf 0.001, "
     "NMS IoU 0.7 (YOLO only), max_det 300, batch 1, FP32. AP = COCO AP@[.5:.95] (pycocotools).", ""]
L += ["## 1. Overall (all GT classes in each dataset)", "",
      "| Model | Dataset | #img | AP | AP50 | AP75 | AP_S | AP_M | AP_L | AR100 | ms/img | FPS | Peak GPU MB | GPU |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for (mod, ds), m in sorted(runs.items()):
    s = m.get("all_classes", {}).get("stats", {})
    mem = m["memory_mb"]["peak_allocated"]
    mem_s = f"{mem:.0f}" if mem else "n/a"
    L.append(f"| {mod} | {ds} | {m['config']['n_images']} | " + " | ".join(pct(s.get(k)) for k in
             ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR100"]) +
             f" | {m['speed_ms']['total']:.1f} | {m['speed_ms']['fps_end_to_end']:.1f} | {mem_s} | {m['env']['gpu']} |")

L += ["", "## 2. Domain shift COCO -> KITTI on shared classes (person, car, truck)", "",
      "| Model | AP COCO | AP KITTI | ΔAP | AP50 COCO | AP50 KITTI | AP_S COCO | AP_S KITTI | AP_M COCO | AP_M KITTI | AP_L COCO | AP_L KITTI |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for mod in models:
    c, k = runs.get((mod, "coco")), runs.get((mod, "kitti"))
    if not (c and k):
        continue
    cs, ks = c["shared_classes"]["stats"], k["shared_classes"]["stats"]
    cells = [pct(cs["AP"]), pct(ks["AP"]), f"{100 * (ks['AP'] - cs['AP']):+.1f}"]
    for key in ["AP50", "AP_small", "AP_medium", "AP_large"]:
        cells += [pct(cs[key]), pct(ks[key])]
    L.append(f"| {mod} | " + " | ".join(cells) + " |")

L += ["", "## 3. Per-class AP, COCO vs KITTI", "",
      "| Model | Class | AP COCO | AP KITTI | ΔAP | AP50 COCO | AP50 KITTI | #GT KITTI |", "|---|---|---|---|---|---|---|---|"]
for mod in models:
    c, k = runs.get((mod, "coco")), runs.get((mod, "kitti"))
    if not (c and k):
        continue
    for cls, kv in k["shared_classes"]["per_class"].items():
        cv = c["shared_classes"]["per_class"][cls]
        d = f"{100 * (kv['AP'] - cv['AP']):+.1f}" if kv["AP"] is not None and cv["AP"] is not None else "n/a"
        L.append(f"| {mod} | {cls} | {pct(cv['AP'])} | {pct(kv['AP'])} | {d} | {pct(cv['AP50'])} | {pct(kv['AP50'])} | {kv['n_gt']} |")

L += ["", "## 4. Latency breakdown (ms/img, batch 1)", "",
      "| Model | Dataset | Preprocess | Inference | Postprocess (NMS) | Total |", "|---|---|---|---|---|---|"]
for (mod, ds), m in sorted(runs.items()):
    s = m["speed_ms"]
    L.append(f"| {mod} | {ds} | {s['preprocess']:.2f} | {s['inference']:.2f} | {s['postprocess']:.2f} | {s['total']:.2f} |")

L += ["", "## 5. Reproducibility", ""]
for (mod, ds), m in sorted(runs.items()):
    L.append(f"- **{mod} / {ds}**: `{m['config']['command']}` | weights md5 `{m['config']['weights_md5']}` | "
             f"{m['env']['gpu']} on {m['env']['host']} (job {m['env']['slurm_job']}) | torch {m['env']['torch']}, "
             f"CUDA {m['env']['cuda']}, ultralytics {m['env']['ultralytics']}")

(RES / "summary.md").write_text("\n".join(L) + "\n")

fig, ax = plt.subplots(figsize=(6, 4))
for ds, mk in [("coco", "o"), ("kitti", "s")]:
    pts = [(m["speed_ms"]["total"], 100 * m["shared_classes"]["stats"]["AP"], mod)
           for (mod, d), m in runs.items() if d == ds and "shared_classes" in m]
    if pts:
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], marker=mk, label=f"{ds} (person/car/truck)")
        for x, y, t in pts:
            ax.annotate(t, (x, y), textcoords="offset points", xytext=(5, 3), fontsize=8)
ax.set_xlabel("End-to-end latency, ms/img (batch 1)"); ax.set_ylabel("AP@[.5:.95] (%)")
ax.set_title("Speed–accuracy tradeoff"); ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(RES / "speed_accuracy.png", dpi=200)
print((RES / "summary.md").read_text())
