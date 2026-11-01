# Results summary

All runs: COCO-pretrained weights, no fine-tuning, imgsz 640, conf 0.001, NMS IoU 0.7 (YOLO only), max_det 300, batch 1, FP32. AP = COCO AP@[.5:.95] (pycocotools).

COCO = val2017 (5,000 img); KITTI = train+val, 7,481 img (all labeled, Ultralytics version), zero-shot, remapped to person/car/truck.

Hardware: NVIDIA GeForce RTX 3060, driver 617.14, Linux-6.18.54.1-microsoft-standard-WSL2-x86_64-with-glibc2.39, torch 2.11.0+cu128, ultralytics 8.4.197.

## 1. Overall (all GT classes in each dataset)

| Model | Dataset | #img | AP | AP50 | AP75 | AP_S | AP_M | AP_L | AR100 | ms/img | FPS | Peak GPU MB | GPU |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rtdetr-l | coco | 5000 | 52.8 | 71.2 | 57.2 | 34.4 | 57.4 | 70.8 | 71.7 | 46.8 | 21.3 | 239 | NVIDIA GeForce RTX 3060 |
| rtdetr-l | kitti | 7481 | 26.1 | 48.5 | 25.2 | 19.1 | 25.3 | 35.9 | 53.2 | 38.7 | 25.9 | 239 | NVIDIA GeForce RTX 3060 |
| yolo11s | coco | 5000 | 45.7 | 62.3 | 49.3 | 27.8 | 50.2 | 62.5 | 61.7 | 14.4 | 69.2 | 111 | NVIDIA GeForce RTX 3060 |
| yolo11s | kitti | 7481 | 22.3 | 43.7 | 20.7 | 13.2 | 20.2 | 32.5 | 41.0 | 14.4 | 69.6 | 83 | NVIDIA GeForce RTX 3060 |
| yolov8s | coco | 5000 | 44.2 | 60.5 | 47.8 | 25.4 | 49.0 | 60.3 | 61.1 | 13.7 | 73.2 | 79 | NVIDIA GeForce RTX 3060 |
| yolov8s | kitti | 7481 | 22.2 | 43.6 | 20.6 | 17.7 | 20.2 | 32.0 | 40.9 | 10.7 | 93.2 | 56 | NVIDIA GeForce RTX 3060 |

## 2. Domain shift COCO -> KITTI on shared classes (person, car, truck)

| Model | AP COCO | AP KITTI | ΔAP | AP50 COCO | AP50 KITTI | AP_S COCO | AP_S KITTI | AP_M COCO | AP_M KITTI | AP_L COCO | AP_L KITTI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| yolov8s | 45.5 | 22.2 | -23.3 | 65.1 | 43.6 | 27.2 | 17.7 | 55.5 | 20.2 | 67.1 | 32.0 |
| yolo11s | 46.4 | 22.3 | -24.1 | 65.8 | 43.7 | 27.9 | 13.2 | 55.7 | 20.2 | 69.7 | 32.5 |
| rtdetr-l | 52.6 | 26.1 | -26.4 | 74.8 | 48.5 | 34.1 | 19.1 | 61.3 | 25.3 | 75.4 | 35.9 |

## 3. Per-class AP, COCO vs KITTI

| Model | Class | AP COCO | AP KITTI | ΔAP | AP50 COCO | AP50 KITTI | #GT KITTI |
|---|---|---|---|---|---|---|---|
| yolov8s | person | 58.1 | 15.6 | -42.6 | 81.3 | 42.8 | 6336 |
| yolov8s | car | 44.2 | 42.1 | -2.1 | 66.3 | 74.7 | 31656 |
| yolov8s | truck | 34.1 | 9.0 | -25.1 | 47.8 | 13.5 | 1094 |
| yolo11s | person | 58.8 | 15.3 | -43.5 | 81.7 | 43.0 | 6336 |
| yolo11s | car | 44.9 | 43.1 | -1.9 | 66.4 | 75.4 | 31656 |
| yolo11s | truck | 35.5 | 8.5 | -27.0 | 49.4 | 12.7 | 1094 |
| rtdetr-l | person | 61.7 | 16.0 | -45.7 | 86.3 | 44.9 | 6336 |
| rtdetr-l | car | 50.1 | 47.3 | -2.8 | 73.7 | 79.1 | 31656 |
| rtdetr-l | truck | 45.8 | 15.0 | -30.8 | 64.3 | 21.5 | 1094 |

## 4. Latency breakdown (ms/img, batch 1)

| Model | Dataset | Preprocess | Inference | Postprocess (NMS for YOLO; top-k only for RT-DETR) | Total |
|---|---|---|---|---|---|
| rtdetr-l | coco | 1.03 | 44.87 | 0.95 | 46.84 |
| rtdetr-l | kitti | 0.80 | 37.14 | 0.72 | 38.66 |
| yolo11s | coco | 0.61 | 12.04 | 1.79 | 14.44 |
| yolo11s | kitti | 0.74 | 11.70 | 1.91 | 14.36 |
| yolov8s | coco | 0.65 | 11.02 | 2.00 | 13.67 |
| yolov8s | kitti | 0.69 | 8.28 | 1.76 | 10.73 |

## 5. Reproducibility

- **rtdetr-l / coco**: `eval_det.py --model rtdetr-l.pt --dataset coco --imgsz 640 --conf 0.001 --iou 0.7 --max-det 300 --device 0` | weights md5 `8e6bae256b16e9623c050979bb47c224` | COCO val2017 | NVIDIA GeForce RTX 3060 on DESKTOP-5GA8ANJ (run 20261031-200219, driver 617.14, Linux-6.18.54.1-microsoft-standard-WSL2-x86_64-with-glibc2.39) | torch 2.11.0+cu128, CUDA 12.8, ultralytics 8.4.197
- **rtdetr-l / kitti**: `eval_det.py --model rtdetr-l.pt --dataset kitti --imgsz 640 --conf 0.001 --iou 0.7 --max-det 300 --device 0` | weights md5 `8e6bae256b16e9623c050979bb47c224` | KITTI train+val (all labeled, Ultralytics version) | NVIDIA GeForce RTX 3060 on DESKTOP-5GA8ANJ (run 20261031-200219, driver 617.14, Linux-6.18.54.1-microsoft-standard-WSL2-x86_64-with-glibc2.39) | torch 2.11.0+cu128, CUDA 12.8, ultralytics 8.4.197
- **yolo11s / coco**: `eval_det.py --model yolo11s.pt --dataset coco --imgsz 640 --conf 0.001 --iou 0.7 --max-det 300 --device 0` | weights md5 `9637097d5fbdc1002d25d2d3d9f7c435` | COCO val2017 | NVIDIA GeForce RTX 3060 on DESKTOP-5GA8ANJ (run 20261031-200219, driver 617.14, Linux-6.18.54.1-microsoft-standard-WSL2-x86_64-with-glibc2.39) | torch 2.11.0+cu128, CUDA 12.8, ultralytics 8.4.197
- **yolo11s / kitti**: `eval_det.py --model yolo11s.pt --dataset kitti --imgsz 640 --conf 0.001 --iou 0.7 --max-det 300 --device 0` | weights md5 `9637097d5fbdc1002d25d2d3d9f7c435` | KITTI train+val (all labeled, Ultralytics version) | NVIDIA GeForce RTX 3060 on DESKTOP-5GA8ANJ (run 20261031-200219, driver 617.14, Linux-6.18.54.1-microsoft-standard-WSL2-x86_64-with-glibc2.39) | torch 2.11.0+cu128, CUDA 12.8, ultralytics 8.4.197
- **yolov8s / coco**: `eval_det.py --model yolov8s.pt --dataset coco --imgsz 640 --conf 0.001 --iou 0.7 --max-det 300 --device 0` | weights md5 `0a1d5d0619d5a24848a267ec1e3a3118` | COCO val2017 | NVIDIA GeForce RTX 3060 on DESKTOP-5GA8ANJ (run 20261031-200219, driver 617.14, Linux-6.18.54.1-microsoft-standard-WSL2-x86_64-with-glibc2.39) | torch 2.11.0+cu128, CUDA 12.8, ultralytics 8.4.197
- **yolov8s / kitti**: `eval_det.py --model yolov8s.pt --dataset kitti --imgsz 640 --conf 0.001 --iou 0.7 --max-det 300 --device 0` | weights md5 `0a1d5d0619d5a24848a267ec1e3a3118` | KITTI train+val (all labeled, Ultralytics version) | NVIDIA GeForce RTX 3060 on DESKTOP-5GA8ANJ (run 20261031-200219, driver 617.14, Linux-6.18.54.1-microsoft-standard-WSL2-x86_64-with-glibc2.39) | torch 2.11.0+cu128, CUDA 12.8, ultralytics 8.4.197
