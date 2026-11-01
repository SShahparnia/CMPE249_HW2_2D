#!/usr/bin/env bash
# Downloads COCO val2017, KITTI (Ultralytics 2D-detection version, ~390 MB), and the 3 checkpoints.
# Data lives on the WSL ext4 disk (fast I/O); $PROJ/data is a symlink to it.
set -euo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_REAL="${CMPE249_DATA:-$HOME/cmpe249_data}"          # WSL ext4, not /mnt/c
mkdir -p "$DATA_REAL/coco" "$PROJ/weights" "$PROJ/results" "$PROJ/logs"
ln -sfn "$DATA_REAL" "$PROJ/data"
source "$PROJ/.venv/bin/activate"

# --- COCO val2017 (5,000 images) + official annotations ---
cd "$DATA_REAL/coco"
wget -nc http://images.cocodataset.org/zips/val2017.zip
wget -nc http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q -n val2017.zip
unzip -q -n annotations_trainval2017.zip

# --- KITTI: Ultralytics downloads into its datasets_dir, so point that at the real ext4 data folder ---
yolo settings datasets_dir="$DATA_REAL"
python "$PROJ/code/make_kitti_gt.py"      # downloads KITTI on first run + writes COCO-format GT json

# --- Checkpoints (COCO-pretrained, official Ultralytics releases) ---
cd "$PROJ/weights"
python - <<'PY'
from ultralytics import YOLO, RTDETR
YOLO("yolov8s.pt"); YOLO("yolo11s.pt"); RTDETR("rtdetr-l.pt")
PY
ls -lh "$PROJ/weights"
echo "Done."
