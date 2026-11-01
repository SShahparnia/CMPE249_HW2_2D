#!/usr/bin/env bash
# Latency reruns on an idle GPU: REPS (default 3) x 3 models x 2 datasets, interleaved by rep so
# slow drift (thermals, background load) spreads across all models instead of biasing one.
# Output: reruns/<RUN_ID>/{rep1,rep2,rep3}/<model>_<dataset>/metrics.json, gpu_state.csv,
#         latency_summary.md, latency_errorbars.png, logs/
# Usage:  bash code/rerun_timing.sh            (REPS=3 MAX_UTIL=10 by default)
set -euo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJ/.venv/bin/activate"
export PYTHONNOUSERSITE=1
REPS="${REPS:-3}"
MAX_UTIL="${MAX_UTIL:-10}"   # abort if the idle GPU is busier than this (%) before starting
export RUN_ID="${RUN_ID:-rerun-$(date +%Y%m%d-%H%M%S)}"
OUT="$PROJ/reruns/$RUN_ID"
mkdir -p "$OUT/logs"; LOG="$OUT/logs/run-$RUN_ID.log"
exec > >(tee -a "$LOG") 2>&1
cd "$PROJ/code"

# Mean GPU utilization (%) and memory used (MiB) over 5 one-second samples
# (nvidia-smi's own -l loop option is not supported under WSL).
gpu_state_safe() {
  for i in 1 2 3 4 5; do
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits; sleep 1
  done | awk -F', ' '{u+=$1; m+=$2; n++} END {printf "%.0f %.0f\n", u/n, m/n}'
}

# ---- Reproducibility header ----
echo "Host: $(hostname)   Run: $RUN_ID   Date: $(date)   Kernel: $(uname -r)   Reps: $REPS"
grep PRETTY_NAME /etc/os-release
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvidia-smi > "$OUT/logs/nvidia-smi-start-$RUN_ID.txt"
python -c "import torch,torchvision,ultralytics;print('torch',torch.__version__,'cuda',torch.version.cuda,'cudnn',torch.backends.cudnn.version(),'tv',torchvision.__version__,'ultralytics',ultralytics.__version__)"
pip freeze > "$OUT/logs/pip-freeze-$RUN_ID.txt"
md5sum "$PROJ"/weights/*.pt | tee "$OUT/logs/weights-md5-$RUN_ID.txt"

# ---- Idle-GPU gate ----
read -r UTIL MEM <<<"$(gpu_state_safe)"
echo "Idle check before start: GPU util ${UTIL}%  memory used ${MEM} MiB (limit ${MAX_UTIL}%)"
if [ "$UTIL" -gt "$MAX_UTIL" ]; then
  echo "ABORT: GPU is ${UTIL}% busy before starting. Close browsers/games/Discord/video and retry,"
  echo "       or override with MAX_UTIL=<pct> bash code/rerun_timing.sh"
  exit 1
fi

echo "rep,model,dataset,util_before_pct,mem_before_mib,time" > "$OUT/gpu_state.csv"
for REP in $(seq 1 "$REPS"); do
  for MODEL in yolov8s.pt yolo11s.pt rtdetr-l.pt; do
    for DATA in coco kitti; do
      read -r UTIL MEM <<<"$(gpu_state_safe)"
      echo "$REP,${MODEL%.pt},$DATA,$UTIL,$MEM,$(date +%T)" >> "$OUT/gpu_state.csv"
      CMD="python eval_det.py --model $MODEL --dataset $DATA --imgsz 640 --conf 0.001 --iou 0.7 --max-det 300 --device 0 --save-vis 0 --out $OUT/rep$REP"
      echo -e "\n>>> rep $REP/$REPS  $(date +%T)  GPU before: ${UTIL}% ${MEM} MiB\n>>> $CMD"
      $CMD
    done
  done
done

nvidia-smi > "$OUT/logs/nvidia-smi-end-$RUN_ID.txt"
python summarize_reruns.py "$OUT"
echo "Done. Log: $LOG"
