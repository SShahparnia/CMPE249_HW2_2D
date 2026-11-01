#!/usr/bin/env python
"""
Builds the report's failure-case figure from saved detections (no inference).
For each failure mode it picks the KITTI image (deterministically, by largest GT box of that kind) that
shows it most clearly for the given model, and draws GT (green, KITTI class) + predictions (red, COCO
class, conf >= 0.25) cropped around the case.

Modes: cyclist (rider-only person box, low IoU) | van called truck | tram called train (dropped GT -> FP)
       | small/distant pedestrian missed
Usage:  python make_report_figures.py [--model rtdetr-l]  -> report/figures/failure_cases.jpg (+ .json)
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics.data.utils import img2label_paths

PROJ = Path(__file__).resolve().parents[1]
DATA = (PROJ / "data").resolve()
GT = json.loads((DATA / "kitti_coco/kitti_trainval_coco.json").read_text())
REPORT = json.loads((DATA / "kitti_coco/mapping_report.json").read_text())
KNAMES = {int(k): v for k, v in REPORT["kitti_names"].items()}
COCO_NAME = {c["id"]: c["name"] for c in json.loads((DATA / "coco/annotations/instances_val2017.json").read_text())["categories"]}
CONF = 0.25


def iou(a, b):
    if len(b) == 0:
        return np.zeros(0)
    iw = np.clip(np.minimum(a[0] + a[2], b[:, 0] + b[:, 2]) - np.maximum(a[0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(a[1] + a[3], b[:, 1] + b[:, 3]) - np.maximum(a[1], b[:, 1]), 0, None)
    inter = iw * ih
    return inter / (a[2] * a[3] + b[:, 2] * b[:, 3] - inter + 1e-9)


ap = argparse.ArgumentParser()
ap.add_argument("--model", default="rtdetr-l")
a = ap.parse_args()

imgs = {im["id"]: im for im in GT["images"]}
gt = defaultdict(list)
for im in GT["images"]:
    lp = Path(img2label_paths([im["file_name"]])[0])
    if not lp.exists():
        continue
    w, h = im["width"], im["height"]
    for line in lp.read_text().splitlines():
        if line.strip():
            c, xc, yc, bw, bh = line.split()[:5]
            bw, bh = float(bw) * w, float(bh) * h
            gt[im["id"]].append((KNAMES[int(c)], np.array([float(xc) * w - bw / 2, float(yc) * h - bh / 2, bw, bh])))
pred = defaultdict(list)
for d in json.loads((PROJ / f"results/{a.model}_kitti/detections.json").read_text()):
    if d["score"] >= CONF:
        pred[d["image_id"]].append((d["category_id"], d["score"], np.array(d["bbox"])))

cands = defaultdict(list)  # mode -> [(sort_key, img_id, box, note)]
for iid, boxes in gt.items():
    P = pred.get(iid, [])
    pc = np.array([p[0] for p in P]); pb = np.array([p[2] for p in P]).reshape(-1, 4)
    for k, b in boxes:
        io = iou(b, pb)
        person = io[pc == 1] if len(pc) else np.zeros(0)
        best = int(pc[io.argmax()]) if len(io) and io.max() >= 0.5 else None
        if k == "cyclist" and b[3] >= 80 and len(person) and 0.25 <= person.max() < 0.5:
            cands["cyclist"].append((b[3], iid, b, f"cyclist GT vs 'person' pred: IoU {person.max():.2f}"))
        if k == "van" and b[3] >= 60 and best == 8:
            cands["van"].append((b[3], iid, b, f"van GT (→car) predicted 'truck', IoU {io.max():.2f}"))
        if k == "tram" and b[3] >= 60 and best == 7:
            cands["tram"].append((b[3], iid, b, f"tram (dropped) predicted 'train' → counts as FP"))
        if k == "pedestrian" and 20 <= b[3] < 32 and (not len(person) or person.max() < 0.5):
            cands["small"].append((-abs(b[3] - 26) - 0.1 * len(P), iid, b, f"pedestrian {b[3]:.0f} px tall, missed"))

TITLES = {"cyclist": "(a) Cyclist box covers rider + bike; model boxes the rider only",
          "van": "(b) KITTI 'van' (mapped to car) predicted as COCO 'truck'",
          "tram": "(c) Tram is dropped from GT, so the confident 'train' box is a false positive",
          "small": "(d) Distant pedestrian (<32 px tall at 1242 px width) missed"}
panels, meta = [], {}
for mode in ("cyclist", "van", "tram", "small"):
    if not cands[mode]:
        continue
    _, iid, box, note = sorted(cands[mode], key=lambda t: -t[0])[0]
    im = imgs[iid]; img = cv2.imread(im["file_name"]); H, W = img.shape[:2]
    fs = 0.45 if mode == "small" else 0.6
    for k, b in gt[iid]:  # GT green; the box this panel is about in thick yellow
        x, y, w, h = map(int, b)
        tgt = np.allclose(b, box)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 220, 255) if tgt else (0, 200, 0), 3 if tgt else 1)
        if tgt:
            cv2.putText(img, f"GT: {k}", (x, min(y + h + 16, H - 4)), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 220, 255), 2, cv2.LINE_AA)
    cw = 300 if mode == "small" else 620  # zoom in on the small case
    cx = int(box[0] + box[2] / 2); x0 = int(np.clip(cx - cw // 2, 0, max(W - cw, 0)))
    nhit = 0
    for c, s, b in sorted(pred.get(iid, []), key=lambda t: -t[1]):  # predictions red; label only those touching the case box
        x, y, w, h = map(int, b)
        hit = iou(box, b[None])[0] > 0.1
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2 if hit else 1)
        if hit:  # stack labels inside the box, kept inside the crop
            t = f"{COCO_NAME.get(c, c)} {s:.2f}"
            tw = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)[0][0]
            tx = int(np.clip(x + 4, x0 + 2, x0 + cw - tw - 4))
            cv2.putText(img, t, (tx, y + 20 + 22 * nhit), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 255), 2, cv2.LINE_AA)
            nhit += 1
    cy = int(box[1] + box[3] / 2); ch = int(cw * 560 / 930)
    y0 = int(np.clip(cy - ch // 2, 0, max(H - ch, 0))) if mode == "small" else 0
    crop = img[y0:y0 + (ch if mode == "small" else H), x0:x0 + cw]
    crop = cv2.resize(crop, (930, 560))  # KITTI heights vary (370-376 px); fixed size so panels tile
    bar = np.full((34, crop.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, TITLES[mode], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 1, cv2.LINE_AA)
    panels.append(np.vstack([bar, crop]))
    meta[mode] = {"image": Path(im["file_name"]).name, "split": im.get("split"), "note": note}

grid = [np.hstack(panels[i:i + 2]) if len(panels[i:i + 2]) == 2 else
        np.hstack([panels[i], np.full_like(panels[i], 255)]) for i in range(0, len(panels), 2)]
out = PROJ / "report/figures"; out.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out / "failure_cases.jpg"), np.vstack(grid), [cv2.IMWRITE_JPEG_QUALITY, 88])
(out / "failure_cases.json").write_text(json.dumps({"model": a.model, "conf": CONF, "cases": meta}, indent=2))
print(json.dumps(meta, indent=2))

# ---- Figure 2: per-class AP, COCO vs KITTI (small multiples: one panel per shared class) ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C_COCO, C_KITTI = "#2a78d6", "#eb6834"  # reference palette categorical slots 1, 2
MODELS = ("yolov8s", "yolo11s", "rtdetr-l")
runs = {}
for f in (PROJ / "results").glob("*/metrics.json"):
    m = json.loads(f.read_text())
    runs[(Path(m["config"]["model"]).stem, m["config"]["dataset"])] = m

plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True, facecolor=SURF)
x, bw = np.arange(len(MODELS)), 0.36
for ax, cls in zip(axes, ("person", "car", "truck")):
    ax.set_facecolor(SURF)
    coco = [100 * runs[(m, "coco")]["shared_classes"]["per_class"][cls]["AP"] for m in MODELS]
    kitti = [100 * runs[(m, "kitti")]["shared_classes"]["per_class"][cls]["AP"] for m in MODELS]
    ax.bar(x - bw / 2 - 0.01, coco, bw, color=C_COCO, label="COCO val2017", zorder=2)
    ax.bar(x + bw / 2 + 0.01, kitti, bw, color=C_KITTI, label="KITTI (zero-shot)", zorder=2)
    for xi, c, k in zip(x, coco, kitti):  # selective direct label: the drop only
        ax.annotate(f"{k - c:+.0f}", (xi + bw / 2, k), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8, color=INK2)
    ax.set_title(cls, fontsize=10, color=INK)
    ax.set_xticks(x, MODELS, fontsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
axes[0].set_ylabel("AP@[.5:.95] (%)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(out / "domain_shift_per_class.png", dpi=220, facecolor=SURF)
print("wrote", out / "domain_shift_per_class.png")

# ---- Figure 3: idle-GPU latency, mean ± std over reruns, stacked by stage ----
rr = sorted((PROJ / "reruns").glob("rerun-*"))[-1]
lat = defaultdict(list)
for f in rr.glob("rep*/*/metrics.json"):
    m = json.loads(f.read_text())
    lat[(Path(m["config"]["model"]).stem, m["config"]["dataset"])].append(m["speed_ms"])
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), sharex=True, facecolor=SURF)
STAGES = (("preprocess", "#86b6ef"), ("inference", "#2a78d6"), ("postprocess", "#eb6834"))
for ax, ds in zip(axes, ("coco", "kitti")):
    ax.set_facecolor(SURF)
    y = np.arange(len(MODELS))[::-1]
    for yi, mod in zip(y, MODELS):
        rs = lat[(mod, ds)]
        left = 0.0
        for st, col in STAGES:
            v = float(np.mean([r[st] for r in rs]))
            ax.barh(yi, v, 0.55, left=left, color=col, edgecolor=SURF, linewidth=1,
                    label=st if (yi == y[0] and ds == "coco") else None, zorder=2)
            left += v
        tot = [r["total"] for r in rs]
        mu, sd = float(np.mean(tot)), float(np.std(tot, ddof=1))
        ax.errorbar(mu, yi, xerr=sd, color=INK, capsize=3, lw=1, zorder=3)
        ax.annotate(f"{mu:.1f} ± {sd:.1f} ms", (mu + sd, yi), xytext=(5, -3), textcoords="offset points",
                    fontsize=8, color=INK2)
    ax.set_yticks(y, MODELS, fontsize=8)
    ax.set_title(f"{'COCO val2017' if ds == 'coco' else 'KITTI train+val'}", fontsize=10, color=INK)
    ax.xaxis.grid(True, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(length=0); ax.set_xlim(0, 55)
    ax.set_xlabel("ms / image (batch 1, mean of 3 idle-GPU reps)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=8, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig(out / "latency_breakdown.png", dpi=220, facecolor=SURF)
print("wrote", out / "latency_breakdown.png", "from", rr.name)
