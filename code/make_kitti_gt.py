#!/usr/bin/env python
"""
Convert the Ultralytics KITTI val split (YOLO txt labels, KITTI classes) into a COCO-format
ground-truth json whose categories use COCO ids, via kitti_to_coco_mapping.json.
This lets COCO-pretrained detectors be scored with the official pycocotools COCOeval, no retraining.

Output: ~/cmpe249/data/kitti_coco/kitti_val_coco.json (+ mapping_report.json)
"""
import json
from collections import Counter
from pathlib import Path

from PIL import Image
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset, img2label_paths

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "data/kitti_coco"
COCO_CATS = {"person": 1, "car": 3, "truck": 8}  # official COCO category ids

mapping = json.loads((Path(__file__).parent / "kitti_to_coco_mapping.json").read_text())
data = check_det_dataset("kitti.yaml", autodownload=True)
names = {int(k): v.lower().replace(" ", "_") for k, v in data["names"].items()}
unknown = [n for n in names.values() if n not in mapping]
assert not unknown, f"Add these KITTI classes to kitti_to_coco_mapping.json: {unknown}"

val = Path(data["val"])
imgs = sorted(p for p in val.rglob("*") if p.suffix[1:].lower() in IMG_FORMATS)
labels = img2label_paths([str(p) for p in imgs])

images, anns, kept, dropped = [], [], Counter(), Counter()
for img_id, (ip, lp) in enumerate(zip(imgs, labels), start=1):
    w, h = Image.open(ip).size
    images.append({"id": img_id, "file_name": str(ip), "width": w, "height": h})
    if not Path(lp).exists():
        continue
    for line in Path(lp).read_text().splitlines():
        if not line.strip():
            continue
        c, xc, yc, bw, bh = line.split()[:5]
        kname = names[int(c)]
        tgt = mapping[kname]
        if tgt is None:
            dropped[kname] += 1
            continue
        bw, bh = float(bw) * w, float(bh) * h
        x, y = float(xc) * w - bw / 2, float(yc) * h - bh / 2
        anns.append({"id": len(anns) + 1, "image_id": img_id, "category_id": COCO_CATS[tgt],
                     "bbox": [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)],
                     "area": round(bw * bh, 2), "iscrowd": 0})
        kept[f"{kname}->{tgt}"] += 1

OUT.mkdir(parents=True, exist_ok=True)
gt = {"info": {"description": "KITTI val (Ultralytics split) remapped to COCO ids"},
      "images": images, "annotations": anns,
      "categories": [{"id": i, "name": n} for n, i in COCO_CATS.items()]}
(OUT / "kitti_val_coco.json").write_text(json.dumps(gt))
report = {"n_images": len(images), "n_annotations": len(anns), "kept": kept, "dropped": dropped,
          "kitti_names": names, "mapping": {k: v for k, v in mapping.items() if not k.startswith("_")}}
(OUT / "mapping_report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
