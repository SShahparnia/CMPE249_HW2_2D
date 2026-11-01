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

Results land in `results/summary.md` (all tables for the report) and `results/speed_accuracy.png`.

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

## Settings (state these in the report)

- imgsz 640, conf 0.001 (standard for mAP), NMS IoU 0.7 (YOLO only; RT-DETR does not use NMS), max_det 300, batch 1, FP32.
- Latency = mean Ultralytics preprocess + inference + postprocess per image (CUDA-synchronized), after 20 warm-up passes.
- The domain-shift comparison uses only the **shared classes** (person, car, truck). COCO is re-scored on just those 3 classes so the comparison is apples-to-apples.

## Caveats to discuss in the report

- **Cyclist → person.** KITTI boxes cover rider + bike, while COCO models box only the rider. Expect low IoU, and person AP75 hurt more than AP50.
- **Dropped classes.** Tram/Misc/DontCare regions are not in the GT, so detections there count as false positives.
- **Scale.** KITTI is 1242×375; at imgsz 640 everything shrinks ~1.9×, so many pedestrians become COCO-"small". Compare AP_S vs AP_L.
- **Area definition.** KITTI "area" is box area; COCO uses mask area, so S/M/L buckets differ slightly.
- **One machine.** RTX 3060 under WSL2. Close GPU apps and use a fixed (High-performance) power plan. WSL2 adds a little latency vs native Linux, but all models share the same conditions.
- **Using KITTI train is valid.** No model saw KITTI during training, so all 7,481 labeled images are unseen test data.

## Suggested report outline (15 slides)

1. Title / goal
2. Models and families: YOLOv8s vs YOLO11s (C2f → C3k2 + C2PSA) vs RT-DETR-L (hybrid encoder + query decoder, NMS-free)
3. Datasets + taxonomy mapping table
4. Setup: hardware, versions, thresholds, exact commands (copy from `summary.md` §5)
5. COCO results (Table 1)
6–7. COCO→KITTI domain shift: overall + per-class / per-scale (Tables 2–3)
8. Speed–accuracy plot
9. Latency breakdown (Table 4: NMS cost in YOLO postprocess vs RT-DETR)
10. Architecture comparison
11–12. Failure analysis from `vis/` images (missed small pedestrians, cyclist box mismatch, vans→truck, FPs on trams)
13. Conclusions
14. References (≥8, ≥4 papers, ≥2 docs)
15. Appendix: repo link, `code/`, `logs/*.log`, `results/*/metrics.json`

Submit the code folder + `logs/` + `results/*/metrics.json` as the command log.
