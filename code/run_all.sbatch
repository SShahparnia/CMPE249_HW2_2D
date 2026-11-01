#!/bin/bash
#SBATCH --job-name=hw1-eval
#SBATCH --output=/home/SJSU_ID/cmpe249/logs/%x-%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
##SBATCH --nodelist=g7             # optional: pin an A100 so every latency number comes from the same GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -eo pipefail
PROJ=/home/SJSU_ID/cmpe249
source ~/miniconda3/etc/profile.d/conda.sh
conda activate det
export PYTHONNOUSERSITE=1
cd $PROJ/code

# ---- Reproducibility header (lands in the log) ----
echo "Node: $(hostname)   Job: $SLURM_JOB_ID   Date: $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch, torchvision, ultralytics; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'tv', torchvision.__version__, 'ultralytics', ultralytics.__version__)"
pip freeze > $PROJ/logs/pip-freeze-$SLURM_JOB_ID.txt

# ---- 3 models (2 families: YOLO CNN + RT-DETR transformer) x 2 datasets ----
for MODEL in yolo11n.pt yolo11m.pt rtdetr-l.pt; do
  for DATA in coco kitti; do
    CMD="python eval_det.py --model $MODEL --dataset $DATA --imgsz 640 --conf 0.001 --iou 0.7 --max-det 300 --device 0"
    echo -e "\n>>> $(date +%T)  $CMD"
    $CMD
  done
done

python summarize.py
