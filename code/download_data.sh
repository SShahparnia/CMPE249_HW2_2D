#!/bin/bash
# Downloads COCO val2017, KITTI (Ultralytics 2D-detection version, ~390 MB), and the 3 checkpoints.
# Run on the LOGIN node (compute nodes may not have internet). Light I/O, no inference.
set -eo pipefail
PROJ=$HOME/cmpe249
mkdir -p $PROJ/data/coco $PROJ/weights $PROJ/results $PROJ/logs
source ~/miniconda3/etc/profile.d/conda.sh
conda activate det

# --- COCO val2017 (5,000 images) + official annotations ---
cd $PROJ/data/coco
wget -nc http://images.cocodataset.org/zips/val2017.zip
wget -nc http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q -n val2017.zip
unzip -q -n annotations_trainval2017.zip

# --- KITTI: Ultralytics downloads into its datasets_dir, so point that at our data folder ---
yolo settings datasets_dir=$PROJ/data
python $PROJ/code/make_kitti_gt.py        # downloads KITTI on first run + writes COCO-format GT json

# --- Checkpoints (COCO-pretrained, official Ultralytics releases) ---
cd $PROJ/weights
python - <<'PY'
from ultralytics import YOLO, RTDETR
from ultralytics.utils.checks import check_font
YOLO("yolo11n.pt"); YOLO("yolo11m.pt"); RTDETR("rtdetr-l.pt")
check_font("Arial.ttf")   # pre-cache so compute nodes never need internet
PY
ls -lh $PROJ/weights
echo "Done."
