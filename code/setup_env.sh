#!/usr/bin/env bash
# One-time local setup (WSL2 Ubuntu). Venv lives on ext4; $PROJ/.venv is a symlink to it.
set -euo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${CMPE249_VENV:-$HOME/.venvs/cmpe249}"
python3 -m venv "$VENV"
ln -sfn "$VENV" "$PROJ/.venv"
source "$PROJ/.venv/bin/activate"
pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics pycocotools matplotlib
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch,ultralytics,pycocotools;print('torch',torch.__version__,'cuda',torch.version.cuda,'gpu_ok',torch.cuda.is_available(),'| ultralytics',ultralytics.__version__)"
