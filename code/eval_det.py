#!/usr/bin/env python
"""
Zero-shot evaluation of a COCO-pretrained Ultralytics detector (YOLOv8, YOLO11 or RT-DETR) on COCO val2017
or KITTI train+val (7,481 images), scored with the official pycocotools COCOeval. No training / fine-tuning.

Reports: AP@[.5:.95], AP50, AP75, AP S/M/L, AR, per-class AP, batch-1 latency/FPS, peak GPU memory.
Saves:   results/<model>_<dataset>/{metrics.json, detections.json, vis/*.jpg}

Example: python eval_det.py --model yolo11s.pt --dataset kitti
"""
import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
import ultralytics
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import RTDETR, YOLO
from ultralytics.data.converter import coco80_to_coco91_class

PROJ = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("CMPE249_DATA", PROJ / "data")).resolve()  # follows data/ symlink to ext4
GT = {"coco": DATA / "coco/annotations/instances_val2017.json",
      "kitti": DATA / "kitti_coco/kitti_trainval_coco.json"}
IMG_ROOT = {"coco": DATA / "coco/val2017", "kitti": None}  # kitti json stores absolute paths
SPLIT = {"coco": "COCO val2017", "kitti": "KITTI train+val (all labeled, Ultralytics version)"}
SHARED_CATS = [1, 3, 8]  # COCO ids person/car/truck = classes that exist in both datasets
STAT_NAMES = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
              "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="yolov8s.pt | yolo11s.pt | rtdetr-l.pt")
    ap.add_argument("--dataset", choices=list(GT), required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001, help="low conf for mAP (standard practice)")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU (ignored by NMS-free RT-DETR)")
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--device", default="0")
    ap.add_argument("--warmup", type=int, default=20, help="warm-up passes excluded from timing")
    ap.add_argument("--limit", type=int, default=0, help="only first N images (smoke test)")
    ap.add_argument("--save-vis", type=int, default=12, help="# images to save with GT(green)/pred(red)")
    ap.add_argument("--vis-conf", type=float, default=0.25)
    ap.add_argument("--out", default=str(PROJ / "results"))
    return ap.parse_args()


def gpu_driver():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def load_model(name):
    w = PROJ / "weights" / name
    path = str(w) if w.exists() else name
    model = RTDETR(path) if "rtdetr" in Path(name).stem.lower() else YOLO(path)
    md5 = hashlib.md5(Path(path).read_bytes()).hexdigest() if Path(path).exists() else None
    return model, path, md5


def coco_eval(coco_gt, det_file, img_ids, cat_ids=None):
    coco_dt = coco_gt.loadRes(str(det_file))
    E = COCOeval(coco_gt, coco_dt, "bbox")
    E.params.imgIds = img_ids
    if cat_ids:
        E.params.catIds = cat_ids
    E.evaluate(); E.accumulate(); E.summarize()
    prec = E.eval["precision"]  # [IoU thr, recall, class, area, maxDet]
    per_class = {}
    for k, cid in enumerate(map(int, E.params.catIds)):  # COCOeval makes these numpy ints
        p, p50 = prec[:, :, k, 0, -1], prec[0, :, k, 0, -1]
        per_class[coco_gt.loadCats(cid)[0]["name"]] = {
            "AP": float(p[p > -1].mean()) if (p > -1).any() else None,
            "AP50": float(p50[p50 > -1].mean()) if (p50 > -1).any() else None,
            "n_gt": len(coco_gt.getAnnIds(imgIds=img_ids, catIds=[cid])),
        }
    return {"stats": dict(zip(STAT_NAMES, map(float, E.stats))), "per_class": per_class}


def draw(r, gt_anns, names, vis_conf, out_path):
    img = r.orig_img.copy()
    for a in gt_anns:  # ground truth = green
        x, y, w, h = map(int, a["bbox"])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 200, 0), 2)
    b = r.boxes
    for (x1, y1, x2, y2), s, c in zip(b.xyxy.tolist(), b.conf.tolist(), b.cls.tolist()):
        if s < vis_conf:
            continue
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(img, f"{names[int(c)]} {s:.2f}", (int(x1), max(int(y1) - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    cv2.imwrite(str(out_path), img)


def main():
    a = parse()
    model, wpath, md5 = load_model(a.model)
    run_dir = Path(a.out) / f"{Path(a.model).stem}_{a.dataset}"
    (run_dir / "vis").mkdir(parents=True, exist_ok=True)

    coco_gt = COCO(str(GT[a.dataset]))
    img_ids = sorted(coco_gt.getImgIds())
    if a.limit:
        img_ids = img_ids[:a.limit]
    infos = coco_gt.loadImgs(img_ids)
    root = IMG_ROOT[a.dataset]
    paths = [str(root / i["file_name"]) if root else i["file_name"] for i in infos]
    cls_map = coco80_to_coco91_class()  # model class index (0-79) -> COCO category id (1-90)
    kw = dict(imgsz=a.imgsz, conf=a.conf, iou=a.iou, max_det=a.max_det, device=a.device, verbose=False)
    cuda = torch.cuda.is_available() and a.device != "cpu"

    for _ in range(a.warmup):
        model.predict(paths[0], **kw)
    if cuda:
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()

    vis_idx = set(np.linspace(0, len(paths) - 1, min(a.save_vis, len(paths))).astype(int).tolist()) if a.save_vis else set()
    dets, speeds = [], []
    t0 = time.perf_counter()
    for n, (img_id, p) in enumerate(zip(img_ids, paths)):
        r = model.predict(p, **kw)[0]  # batch size 1
        speeds.append([r.speed["preprocess"], r.speed["inference"], r.speed["postprocess"]])
        b = r.boxes
        for (x1, y1, x2, y2), s, c in zip(b.xyxy.tolist(), b.conf.tolist(), b.cls.tolist()):
            dets.append({"image_id": img_id, "category_id": cls_map[int(c)],
                         "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                         "score": round(s, 5)})
        if n in vis_idx:
            gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img_id], catIds=SHARED_CATS if a.dataset == "kitti" else []))
            draw(r, gt_anns, model.names, a.vis_conf, run_dir / "vis" / f"{Path(p).stem}.jpg")
        if n % 500 == 0:
            print(f"[{n}/{len(paths)}] {p}", flush=True)
    wall = time.perf_counter() - t0

    det_file = run_dir / "detections.json"
    det_file.write_text(json.dumps(dets))
    sp = np.array(speeds)
    total_ms = float(sp.sum(1).mean())
    metrics = {
        "config": {"model": a.model, "weights": wpath, "weights_md5": md5, "dataset": a.dataset,
                   "split": SPLIT[a.dataset],
                   "gt_file": str(GT[a.dataset]), "n_images": len(img_ids), "imgsz": a.imgsz,
                   "conf": a.conf, "nms_iou": a.iou, "max_det": a.max_det, "batch": 1,
                   "precision": "fp32", "warmup": a.warmup, "command": " ".join(sys.argv)},
        "env": {"host": socket.gethostname(), "run_id": os.environ.get("RUN_ID"), "os": platform.platform(),
                "gpu": torch.cuda.get_device_name(0) if cuda else "cpu",
                "gpu_mem_total_mb": torch.cuda.get_device_properties(0).total_memory / 2**20 if cuda else None,
                "driver": gpu_driver(), "python": platform.python_version(), "torch": torch.__version__,
                "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                "torchvision": torchvision.__version__, "ultralytics": ultralytics.__version__},
        "speed_ms": {"preprocess": float(sp[:, 0].mean()), "inference": float(sp[:, 1].mean()),
                     "postprocess": float(sp[:, 2].mean()), "total": total_ms,
                     "fps_end_to_end": 1000 / total_ms, "fps_inference_only": 1000 / float(sp[:, 1].mean()),
                     "wall_clock_s": wall},
        "memory_mb": {"peak_allocated": torch.cuda.max_memory_allocated() / 2**20 if cuda else None,
                      "peak_reserved": torch.cuda.max_memory_reserved() / 2**20 if cuda else None},
        "n_detections": len(dets),
    }
    if dets:
        print("\n=== All GT classes ===")
        metrics["all_classes"] = coco_eval(coco_gt, det_file, img_ids)
        print("\n=== Shared classes (person, car, truck) ===")
        metrics["shared_classes"] = coco_eval(coco_gt, det_file, img_ids, SHARED_CATS)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ("config", "env", "speed_ms", "memory_mb")}, indent=2))
    print(f"Saved -> {run_dir}")


if __name__ == "__main__":
    main()
