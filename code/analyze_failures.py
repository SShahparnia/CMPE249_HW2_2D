#!/usr/bin/env python
"""
Quantitative failure analysis on KITTI, using the saved detections from the main run.

Goes back to the original KITTI labels (so every GT box keeps its KITTI class, including the dropped
tram/misc boxes) and, at a deployment operating point (conf >= 0.25), measures per KITTI class:
  - recall@0.5 / recall@0.75 against predictions of the mapped COCO class
  - best-IoU distribution (median) -> shows the cyclist box-extent mismatch
  - for van/truck GT: which COCO class the best-overlapping prediction has (car vs truck vs bus)
  - for dropped tram/misc boxes: how often a confident prediction sits on them (would-be false positives)
  - recall by GT box height (KITTI's own easy/moderate/hard size cut-offs: 40 / 25 px)
Matching is per-GT "is there any prediction with IoU >= t" (not one-to-one), which is the right
question for a diagnostic, not for AP.

Usage:  python analyze_failures.py            -> results/failure_analysis.{json,md}
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from ultralytics.data.utils import img2label_paths

PROJ = Path(__file__).resolve().parents[1]
DATA = (PROJ / "data").resolve()
RES = PROJ / "results"
GT = json.loads((DATA / "kitti_coco/kitti_trainval_coco.json").read_text())
REPORT = json.loads((DATA / "kitti_coco/mapping_report.json").read_text())
KNAMES = {int(k): v for k, v in REPORT["kitti_names"].items()}
MAPPING = REPORT["mapping"]
COCO_ID = {"person": 1, "car": 3, "truck": 8}
COCO_NAME = {c["id"]: c["name"] for c in json.loads((DATA / "coco/annotations/instances_val2017.json").read_text())["categories"]}
CONF = 0.25
MODELS = ("yolov8s", "yolo11s", "rtdetr-l")


def iou(a, b):  # a: (4,) xywh, b: (N,4) xywh
    if len(b) == 0:
        return np.zeros(0)
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(a[0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(a[1], b[:, 1]), 0, None)
    inter = iw * ih
    return inter / (a[2] * a[3] + b[:, 2] * b[:, 3] - inter + 1e-9)


# ---- every original KITTI box (incl. dropped classes), keyed by GT image id ----
gt_boxes = defaultdict(list)  # img_id -> [(kitti_class, xywh)]
for im in GT["images"]:
    lp = Path(img2label_paths([im["file_name"]])[0])
    if not lp.exists():
        continue
    w, h = im["width"], im["height"]
    for line in lp.read_text().splitlines():
        if not line.strip():
            continue
        c, xc, yc, bw, bh = line.split()[:5]
        bw, bh = float(bw) * w, float(bh) * h
        gt_boxes[im["id"]].append((KNAMES[int(c)], np.array([float(xc) * w - bw / 2, float(yc) * h - bh / 2, bw, bh])))

out = {"operating_conf": CONF, "models": {}}
for mod in MODELS:
    dets = json.loads((RES / f"{mod}_kitti/detections.json").read_text())
    pred = defaultdict(list)
    for d in dets:
        if d["score"] >= CONF:
            pred[d["image_id"]].append((d["category_id"], d["bbox"]))
    stats = defaultdict(lambda: {"n": 0, "r50": 0, "r75": 0, "best_iou": []})
    size = defaultdict(lambda: {"n": 0, "r50": 0})
    best_cls = defaultdict(Counter)  # kitti class -> COCO class of best-overlap pred (IoU>=0.5, any class)
    dropped = defaultdict(lambda: {"n": 0, "covered": 0, "by": Counter()})
    for img_id, boxes in gt_boxes.items():
        P = pred.get(img_id, [])
        pc = np.array([p[0] for p in P]) if P else np.zeros(0, int)
        pb = np.array([p[1] for p in P], float) if P else np.zeros((0, 4))
        for kname, box in boxes:
            ious_all = iou(box, pb)
            tgt = MAPPING.get(kname)
            if tgt is None:  # tram / misc: dropped from GT -> any confident overlap becomes an FP
                dropped[kname]["n"] += 1
                if (ious_all >= 0.5).any():
                    dropped[kname]["covered"] += 1
                    dropped[kname]["by"][COCO_NAME.get(int(pc[ious_all.argmax()]), str(int(pc[ious_all.argmax()])))] += 1
                continue
            m = pc == COCO_ID[tgt]
            ious = ious_all[m]
            b = float(ious.max()) if len(ious) else 0.0
            s = stats[kname]
            s["n"] += 1; s["r50"] += b >= 0.5; s["r75"] += b >= 0.75; s["best_iou"].append(b)
            if (ious_all >= 0.5).any():
                best_cls[kname][COCO_NAME.get(int(pc[ious_all.argmax()]), str(int(pc[ious_all.argmax()])))] += 1
            else:
                best_cls[kname]["(missed)"] += 1
            if tgt == "person":  # KITTI difficulty uses box height: >=40 easy, 25-40 moderate, <25 hard
                hb = "h>=40px" if box[3] >= 40 else "25<=h<40px" if box[3] >= 25 else "h<25px"
                size[hb]["n"] += 1; size[hb]["r50"] += b >= 0.5
    out["models"][mod] = {
        "per_kitti_class": {k: {"n": v["n"], "recall50": v["r50"] / v["n"], "recall75": v["r75"] / v["n"],
                                "median_best_iou": float(np.median(v["best_iou"]))} for k, v in stats.items()},
        "best_overlap_class": {k: dict(v.most_common()) for k, v in best_cls.items()},
        "person_recall50_by_height": {k: {"n": v["n"], "recall50": v["r50"] / v["n"]} for k, v in size.items()},
        "dropped_boxes_covered": {k: {"n": v["n"], "frac_with_conf_pred": v["covered"] / v["n"], "by": dict(v["by"])}
                                  for k, v in dropped.items()},
    }

(RES / "failure_analysis.json").write_text(json.dumps(out, indent=2))

# ---- markdown tables ----
L = ["# KITTI failure analysis (operating point conf ≥ 0.25)", "",
     "Per-GT matching: a GT box counts as found if any prediction of its mapped COCO class overlaps it with "
     "IoU ≥ t (not one-to-one; diagnostic only).", "",
     "## 1. Recall and box overlap by original KITTI class", "",
     "| Model | KITTI class → COCO | #GT | Recall@0.5 | Recall@0.75 | Median best IoU |", "|---|---|---|---|---|---|"]
order = ("pedestrian", "person_sitting", "cyclist", "car", "van", "truck")
for mod in MODELS:
    pc = out["models"][mod]["per_kitti_class"]
    for k in order:
        if k in pc:
            v = pc[k]
            L.append(f"| {mod} | {k} → {MAPPING[k]} | {v['n']} | {100 * v['recall50']:.1f} | {100 * v['recall75']:.1f} | {v['median_best_iou']:.2f} |")
L += ["", "## 2. What the best-overlapping confident prediction is called (IoU ≥ 0.5, any COCO class)", "",
      "| Model | KITTI class | Distribution |", "|---|---|---|"]
for mod in MODELS:
    for k in ("van", "truck", "cyclist"):
        d = out["models"][mod]["best_overlap_class"].get(k, {})
        n = sum(d.values()) or 1
        L.append(f"| {mod} | {k} | " + ", ".join(f"{c} {100 * x / n:.0f}%" for c, x in d.items()) + " |")
L += ["", "## 3. Person recall@0.5 by GT box height (KITTI difficulty cut-offs)", "",
      "| Model | h ≥ 40 px | 25 ≤ h < 40 px | h < 25 px |", "|---|---|---|---|"]
for mod in MODELS:
    s = out["models"][mod]["person_recall50_by_height"]
    L.append(f"| {mod} | " + " | ".join(f"{100 * s[k]['recall50']:.1f} (n={s[k]['n']})" if k in s else "n/a"
                                         for k in ("h>=40px", "25<=h<40px", "h<25px")) + " |")
L += ["", "## 4. Dropped KITTI classes covered by a confident prediction (these become false positives)", "",
      "| Model | Class | #boxes | % with a conf ≥ 0.25 prediction at IoU ≥ 0.5 | Predicted as |", "|---|---|---|---|---|"]
for mod in MODELS:
    for k, v in out["models"][mod]["dropped_boxes_covered"].items():
        L.append(f"| {mod} | {k} | {v['n']} | {100 * v['frac_with_conf_pred']:.1f} | " +
                 ", ".join(f"{c} {x}" for c, x in sorted(v["by"].items(), key=lambda t: -t[1])) + " |")
(RES / "failure_analysis.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
