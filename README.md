# CMPE 249 HW1 — Track A: zero-shot COCO → KITTI detection

Zero-shot evaluation only, no training or fine-tuning. Everything runs **locally on one machine**
(RTX 3060 12 GB, WSL2 Ubuntu 24.04, Python 3.12 venv). No HPC, SLURM or conda.

| Requirement | How this meets it |
|---|---|
| ≥3 configs, ≥2 families | `yolov8s`, `yolo11s` (YOLO: CNN, anchor-free, NMS) and `rtdetr-l` (DETR family: hybrid CNN + transformer, query-based, NMS-free) |
| ≥2 datasets | COCO val2017 (5,000 img) and KITTI train+val (7,481 img, all labeled, Ultralytics version, auto-download, no registration) |
| Label mapping documented | `code/kitti_to_coco_mapping.json` + `results/mapping_report.json` (kept/dropped counts per class and split) |
| Metrics | pycocotools AP, AP50, AP75, AP S/M/L, AR, per-class AP, batch-1 pre/infer/post latency, FPS, peak GPU memory |
| Reproducibility | each `metrics.json` stores the command, thresholds, imgsz, weights md5, GPU, driver, OS, run id and versions; `logs/` holds the run log, `nvidia-smi`, `pip freeze` and weight md5s |
| 2 analyses | (1) cross-dataset generalization COCO→KITTI by class and scale; (2) speed–accuracy / YOLOv8 vs YOLO11 vs RT-DETR architecture comparison |
| Failure analysis | `results/*_kitti/vis/*.jpg` (GT green, predictions red, conf ≥ 0.25) |

## Run order (local, WSL2)

```bash
cd /mnt/c/Users/sherv/Desktop/CMPE249_HW_2D

# 1. One-time setup (~5 min). Venv lives on ext4 (~/.venvs/cmpe249); .venv is a symlink to it.
sudo apt update && sudo apt install -y python3.12-venv
bash code/setup_env.sh                     # torch cu128, ultralytics, pycocotools, matplotlib

# 2. Data + weights (~1.5 GB). Data lives on ext4 (~/cmpe249_data); data/ is a symlink to it.
bash code/download_data.sh                 # COCO val2017, KITTI, 3 checkpoints, builds KITTI GT json

# 3. Optional smoke test (50 images per run, ~2 min)
source .venv/bin/activate && cd code
for M in yolov8s.pt yolo11s.pt rtdetr-l.pt; do for D in coco kitti; do
  python eval_det.py --model $M --dataset $D --limit 50 --warmup 3 --out ../results_smoke
done; done
cd ..

# 4. Full run: 6 evaluations + summary (~40–60 min). Close other GPU apps first.
bash code/run_all.sh                       # log: logs/run-<RUN_ID>.log
```

Results land in `results/summary.md` and `results/speed_accuracy.png`.

## Files

| File | What it does |
|---|---|
| `code/setup_env.sh` | creates the Python venv (torch cu128, ultralytics, pycocotools, matplotlib) and checks the GPU |
| `code/download_data.sh` | gets COCO val2017, KITTI and the 3 weights, then runs `make_kitti_gt.py` |
| `code/make_kitti_gt.py` | converts KITTI train+val YOLO labels (7,481 img) → COCO-format GT json with COCO category ids |
| `code/kitti_to_coco_mapping.json` | car,van→car · truck→truck · pedestrian,person_sitting,cyclist→person · tram,misc dropped |
| `code/eval_det.py` | batch-1 inference, writes COCO-format detections, runs COCOeval, records timing, memory and env, saves visualizations |
| `code/run_all.sh` | 3 models × 2 datasets + `summarize.py`, with a reproducibility header in the log |
| `code/summarize.py` | builds `results/summary.md` tables and the speed–accuracy plot |


