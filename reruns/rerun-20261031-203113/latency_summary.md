# Latency reruns on an idle GPU

3 repetitions of every model × dataset (interleaved by rep), batch 1, FP32, imgsz 640, conf 0.001, NMS IoU 0.7 (YOLO only), max_det 300, 20 warm-up passes. Each repetition times every image in the dataset; values are **mean ± sample std across repetitions** (ms/img).

Hardware: NVIDIA GeForce RTX 3060, driver 617.14, Linux-6.18.54.1-microsoft-standard-WSL2-x86_64-with-glibc2.39, torch 2.11.0+cu128, ultralytics 8.4.197. Run id `rerun-20261031-203113`.

GPU state sampled before each of the 18 evaluations (5 × 1 s mean): utilization 0–6% (mean 1.7%), memory used 1042–1105 MiB.

## 1. Latency, mean ± std over repetitions (ms/img)

| Model | Dataset | n | Preprocess | Inference | Postprocess | Total | FPS (end-to-end) |
|---|---|---|---|---|---|---|---|
| yolov8s | coco | 3 | 0.42 ± 0.04 | 9.01 ± 1.64 | 1.45 ± 0.26 | 10.88 ± 1.94 | 93.7 ± 15.3 |
| yolov8s | kitti | 3 | 0.53 ± 0.04 | 8.05 ± 1.23 | 1.43 ± 0.18 | 10.00 ± 1.45 | 101.3 ± 13.6 |
| yolo11s | coco | 3 | 0.41 ± 0.02 | 10.24 ± 0.22 | 1.29 ± 0.04 | 11.95 ± 0.27 | 83.7 ± 1.9 |
| yolo11s | kitti | 3 | 0.51 ± 0.02 | 9.33 ± 0.41 | 1.29 ± 0.06 | 11.13 ± 0.49 | 90.0 ± 3.9 |
| rtdetr-l | coco | 3 | 0.83 ± 0.03 | 37.33 ± 1.13 | 0.70 ± 0.03 | 38.86 ± 1.19 | 25.8 ± 0.8 |
| rtdetr-l | kitti | 3 | 0.78 ± 0.00 | 36.80 ± 0.19 | 0.68 ± 0.01 | 38.27 ± 0.20 | 26.1 ± 0.1 |

## 2. Is the YOLOv8s vs YOLO11s gap real?

Δ = YOLO11s − YOLOv8s total latency. SE = √(s₁²/n₁ + s₂²/n₂). |Δ| > 2·SE is treated as a real difference (≈95%); otherwise it is within run-to-run noise.

| Dataset | YOLOv8s total | YOLO11s total | Δ (ms) | 2·SE (ms) | Verdict |
|---|---|---|---|---|---|
| coco | 10.88 ± 1.94 | 11.95 ± 0.27 | +1.06 | 2.27 | within noise |
| kitti | 10.00 ± 1.45 | 11.13 ± 0.49 | +1.12 | 1.77 | within noise |

## 3. Original run (busy GPU, ~35% util / ~2 GB from other apps) vs idle reruns

| Model | Dataset | Original total | Rerun total (mean ± std) | Change |
|---|---|---|---|---|
| yolov8s | coco | 13.67 | 10.88 ± 1.94 | -2.78 (-20%) |
| yolov8s | kitti | 10.73 | 10.00 ± 1.45 | -0.72 (-7%) |
| yolo11s | coco | 14.44 | 11.95 ± 0.27 | -2.49 (-17%) |
| yolo11s | kitti | 14.36 | 11.13 ± 0.49 | -3.23 (-23%) |
| rtdetr-l | coco | 46.84 | 38.86 ± 1.19 | -7.99 (-17%) |
| rtdetr-l | kitti | 38.66 | 38.27 ± 0.20 | -0.39 (-1%) |

## 4. Accuracy sanity check (should be identical across reps and to the original run)

| Model | Dataset | Shared-class AP per rep | Original |
|---|---|---|---|
| yolov8s | coco | 45.50, 45.50, 45.50 | 45.50 |
| yolov8s | kitti | 22.24, 22.24, 22.24 | 22.24 |
| yolo11s | coco | 46.41, 46.41, 46.41 | 46.41 |
| yolo11s | kitti | 22.29, 22.29, 22.29 | 22.29 |
| rtdetr-l | coco | 52.55, 52.55, 52.55 | 52.55 |
| rtdetr-l | kitti | 26.12, 26.12, 26.12 | 26.12 |

## 5. Commands

- **rtdetr-l / coco**: `eval_det.py --model rtdetr-l.pt --dataset coco --imgsz 640 --conf 0.001 --iou 0.7 --max-det 300 --device 0 --save-vis 0 --out /mnt/c/Users/sherv/Desktop/CMPE249_HW_2D/reruns/rerun-20261031-203113/rep1` (× 3, `--out` differs per rep) | weights md5 `8e6bae256b16e9623c050979bb47c224`
- **rtdetr-l / kitti**: `eval_det.py --model rtdetr-l.pt --dataset kitti --imgsz 640 --conf 0.001 --iou 0.7 --max-det 300 --device 0 --save-vis 0 --out /mnt/c/Users/sherv/Desktop/CMPE249_HW_2D/reruns/rerun-20261031-203113/rep1` (× 3, `--out` differs per rep) | weights md5 `8e6bae256b16e9623c050979bb47c224`
- **yolo11s / coco**: `eval_det.py --model yolo11s.pt --dataset coco --imgsz 640 --conf 0.001 --iou 0.7 --max-det 300 --device 0 --save-vis 0 --out /mnt/c/Users/sherv/Desktop/CMPE249_HW_2D/reruns/rerun-20261031-203113/rep1` (× 3, `--out` differs per rep) | weights md5 `9637097d5fbdc1002d25d2d3d9f7c435`
- **yolo11s / kitti**: `eval_det.py --model yolo11s.pt --dataset kitti --imgsz 640 --conf 0.001 --iou 0.7 --max-det 300 --device 0 --save-vis 0 --out /mnt/c/Users/sherv/Desktop/CMPE249_HW_2D/reruns/rerun-20261031-203113/rep1` (× 3, `--out` differs per rep) | weights md5 `9637097d5fbdc1002d25d2d3d9f7c435`
- **yolov8s / coco**: `eval_det.py --model yolov8s.pt --dataset coco --imgsz 640 --conf 0.001 --iou 0.7 --max-det 300 --device 0 --save-vis 0 --out /mnt/c/Users/sherv/Desktop/CMPE249_HW_2D/reruns/rerun-20261031-203113/rep1` (× 3, `--out` differs per rep) | weights md5 `0a1d5d0619d5a24848a267ec1e3a3118`
- **yolov8s / kitti**: `eval_det.py --model yolov8s.pt --dataset kitti --imgsz 640 --conf 0.001 --iou 0.7 --max-det 300 --device 0 --save-vis 0 --out /mnt/c/Users/sherv/Desktop/CMPE249_HW_2D/reruns/rerun-20261031-203113/rep1` (× 3, `--out` differs per rep) | weights md5 `0a1d5d0619d5a24848a267ec1e3a3118`
