#!/usr/bin/env bash
# Local run: 3 models x 2 datasets + summary. Usage: bash code/run_all.sh
set -euo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJ/.venv/bin/activate"
export PYTHONNOUSERSITE=1
export RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$PROJ/logs"; LOG="$PROJ/logs/run-$RUN_ID.log"
exec > >(tee -a "$LOG") 2>&1
cd "$PROJ/code"

# ---- Reproducibility header ----
echo "Host: $(hostname)   Run: $RUN_ID   Date: $(date)   Kernel: $(uname -r)"
grep PRETTY_NAME /etc/os-release
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvidia-smi > "$PROJ/logs/nvidia-smi-$RUN_ID.txt"
python -c "import torch,torchvision,ultralytics;print('torch',torch.__version__,'cuda',torch.version.cuda,'cudnn',torch.backends.cudnn.version(),'tv',torchvision.__version__,'ultralytics',ultralytics.__version__)"
pip freeze > "$PROJ/logs/pip-freeze-$RUN_ID.txt"
md5sum "$PROJ"/weights/*.pt | tee "$PROJ/logs/weights-md5-$RUN_ID.txt"

for MODEL in yolov8s.pt yolo11s.pt rtdetr-l.pt; do
  for DATA in coco kitti; do
    CMD="python eval_det.py --model $MODEL --dataset $DATA --imgsz 640 --conf 0.001 --iou 0.7 --max-det 300 --device 0"
    echo -e "\n>>> $(date +%T)  $CMD"
    $CMD
  done
done
python summarize.py
cp "$PROJ/data/kitti_coco/mapping_report.json" "$PROJ/results/"
nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.sm --format=csv   # end-of-run state
echo "Done. Log: $LOG"
