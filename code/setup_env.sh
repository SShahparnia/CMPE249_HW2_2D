#!/bin/bash
# One-time environment setup. Run on the LOGIN node (needs internet). Assumes Miniconda is installed (hpc.md §4).
set -eo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n det python=3.11 -y
conda activate det
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install ultralytics pycocotools matplotlib
python -c "import torch, ultralytics, pycocotools; print('torch', torch.__version__, '| ultralytics', ultralytics.__version__)"
