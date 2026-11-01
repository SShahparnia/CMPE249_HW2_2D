# KITTI failure analysis (operating point conf ≥ 0.25)

Per-GT matching: a GT box counts as found if any prediction of its mapped COCO class overlaps it with IoU ≥ t (not one-to-one; diagnostic only).

## 1. Recall and box overlap by original KITTI class

| Model | KITTI class → COCO | #GT | Recall@0.5 | Recall@0.75 | Median best IoU |
|---|---|---|---|---|---|
| yolov8s | pedestrian → person | 4487 | 60.0 | 23.4 | 0.59 |
| yolov8s | person_sitting → person | 222 | 34.2 | 7.7 | 0.37 |
| yolov8s | cyclist → person | 1627 | 29.0 | 3.7 | 0.34 |
| yolov8s | car → car | 28742 | 80.6 | 53.0 | 0.77 |
| yolov8s | van → car | 2914 | 44.2 | 28.5 | 0.35 |
| yolov8s | truck → truck | 1094 | 32.7 | 28.7 | 0.00 |
| yolo11s | pedestrian → person | 4487 | 58.4 | 22.7 | 0.58 |
| yolo11s | person_sitting → person | 222 | 31.5 | 10.8 | 0.37 |
| yolo11s | cyclist → person | 1627 | 27.5 | 3.3 | 0.31 |
| yolo11s | car → car | 28742 | 80.3 | 53.7 | 0.77 |
| yolo11s | van → car | 2914 | 44.7 | 30.9 | 0.37 |
| yolo11s | truck → truck | 1094 | 31.7 | 27.1 | 0.00 |
| rtdetr-l | pedestrian → person | 4487 | 66.3 | 25.4 | 0.62 |
| rtdetr-l | person_sitting → person | 222 | 45.0 | 12.6 | 0.46 |
| rtdetr-l | cyclist → person | 1627 | 36.3 | 4.9 | 0.42 |
| rtdetr-l | car → car | 28742 | 88.0 | 60.4 | 0.81 |
| rtdetr-l | van → car | 2914 | 65.3 | 46.3 | 0.72 |
| rtdetr-l | truck → truck | 1094 | 66.2 | 56.2 | 0.81 |

## 2. What the best-overlapping confident prediction is called (IoU ≥ 0.5, any COCO class)

| Model | KITTI class | Distribution |
|---|---|---|
| yolov8s | van | car 39%, (missed) 28%, truck 26%, bus 7%, train 0%, parking meter 0% |
| yolov8s | truck | (missed) 36%, truck 30%, train 18%, bus 15%, car 1% |
| yolov8s | cyclist | (missed) 67%, person 29%, bicycle 4%, motorcycle 1%, car 0% |
| yolo11s | van | car 40%, (missed) 29%, truck 25%, bus 5%, train 0%, traffic light 0%, potted plant 0% |
| yolo11s | truck | (missed) 34%, truck 29%, train 18%, bus 17%, car 1% |
| yolo11s | cyclist | (missed) 69%, person 27%, bicycle 3%, motorcycle 1%, car 0%, horse 0% |
| rtdetr-l | van | truck 48%, car 32%, (missed) 16%, bus 4% |
| rtdetr-l | truck | truck 59%, (missed) 17%, train 14%, bus 9%, car 2% |
| rtdetr-l | cyclist | (missed) 60%, person 36%, bicycle 3%, car 1%, motorcycle 0%, parking meter 0%, truck 0% |

## 3. Person recall@0.5 by GT box height (KITTI difficulty cut-offs)

| Model | h ≥ 40 px | 25 ≤ h < 40 px | h < 25 px |
|---|---|---|---|
| yolov8s | 57.9 (n=5463) | 11.0 (n=693) | 1.1 (n=180) |
| yolo11s | 56.3 (n=5463) | 9.4 (n=693) | 0.0 (n=180) |
| rtdetr-l | 63.3 (n=5463) | 26.7 (n=693) | 11.7 (n=180) |

## 4. Dropped KITTI classes covered by a confident prediction (these become false positives)

| Model | Class | #boxes | % with a conf ≥ 0.25 prediction at IoU ≥ 0.5 | Predicted as |
|---|---|---|---|---|
| yolov8s | misc | 973 | 46.0 | truck 175, car 105, motorcycle 82, bus 65, person 11, bicycle 7, chair 2, train 1 |
| yolov8s | tram | 511 | 36.2 | train 149, bus 33, car 3 |
| yolo11s | misc | 973 | 44.0 | truck 195, car 87, motorcycle 72, bus 57, person 11, bench 4, fire hydrant 1, bicycle 1 |
| yolo11s | tram | 511 | 39.9 | train 168, bus 33, car 3 |
| rtdetr-l | misc | 973 | 64.7 | truck 303, car 104, motorcycle 104, bus 78, person 23, bicycle 17, bench 1 |
| rtdetr-l | tram | 511 | 53.6 | train 183, bus 61, truck 30 |
