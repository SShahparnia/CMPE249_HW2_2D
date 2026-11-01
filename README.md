# CMPE 249 HW1 — Track A (minimum-effort version)

Zero-shot evaluation only, no training or fine-tuning.

| Requirement | How this meets it |
|---|---|
| ≥3 configs, ≥2 families | `yolo11n`, `yolo11m` (YOLO, CNN, anchor-free + NMS) and `rtdetr-l` (DETR family, transformer, NMS-free) |
| ≥2 datasets | COCO val2017 (5,000 img), KITTI val (1,496 img, Ultralytics split, auto-download, no registration) |
| Label mapping documented | `code/kitti_to_coco_mapping.json` + `data/kitti_coco/mapping_report.json` (counts kept/dropped) |
| Metrics | pycocotools AP, AP50, AP75, AP S/M/L, AR, per-class AP, batch-1 latency/FPS, peak GPU memory |
| Reproducibility | each `metrics.json` stores the command, thresholds, imgsz, weights md5, GPU, host, SLURM job, versions; the log has `nvidia-smi` + `pip freeze` |
| 2 analyses | (1) cross-dataset generalization COCO→KITTI by class and scale; (2) speed–accuracy / YOLO vs RT-DETR architecture comparison |
| Failure analysis | `results/*_kitti/vis/*.jpg` (GT green, predictions red, conf ≥ 0.25) |

## Run order

```bash
# 0. From your laptop: copy this folder to the HPC
rsync -avP cmpe249/ sjsu-hpc:~/cmpe249/

# 1. On the login node (needs internet; nothing heavy runs here)
cd ~/cmpe249/code
bash setup_env.sh                          # skip if the 'det' env already exists
bash download_data.sh                      # COCO val + KITTI + 3 checkpoints, ~1.5 GB
sed -i "s/SJSU_ID/$USER/g" run_all.sbatch  # fill in your home path

# 2. Optional smoke test on a GPU node (~2 min)
srun -p gpu --gres=gpu -c 4 --mem=16G --time=00:20:00 --pty /bin/bash
conda activate det && cd ~/cmpe249/code
python eval_det.py --model yolo11n.pt --dataset kitti --limit 50 --warmup 3
exit

# 3. Full run: 6 evaluations + summary (~1 h on an A100)
sbatch run_all.sbatch
squeue -u $USER
tail -f ~/cmpe249/logs/hw1-eval-<JOBID>.log
```

Results land in `results/summary.md` (all tables for the report) and `results/speed_accuracy.png`.

## Files

| File | What it does |
|---|---|
| `code/setup_env.sh` | conda env `det` (torch cu126, ultralytics, pycocotools) |
| `code/download_data.sh` | gets data and weights, then runs `make_kitti_gt.py` |
| `code/make_kitti_gt.py` | converts KITTI val YOLO labels → COCO-format GT json with COCO category ids |
| `code/kitti_to_coco_mapping.json` | car,van→car · truck→truck · pedestrian,person_sitting,cyclist→person · tram,misc dropped |
| `code/eval_det.py` | batch-1 inference, writes COCO-format detections, runs COCOeval, records timing and memory, saves visualizations |
| `code/run_all.sbatch` | 3 models × 2 datasets + `summarize.py` |
| `code/summarize.py` | builds `results/summary.md` tables and the speed–accuracy plot |

## Settings (state these in the report)

- imgsz 640, conf 0.001 (standard for mAP), NMS IoU 0.7 (YOLO only; RT-DETR does not use NMS), max_det 300, batch 1, FP32.
- Latency = mean Ultralytics preprocess + inference + postprocess per image (CUDA-synchronized), after 20 warm-up passes.
- The domain-shift comparison uses only the **shared classes** (person, car, truck). COCO is re-scored on just those 3 classes so the comparison is apples-to-apples.

## Caveats to discuss in the report (these are free analysis points)

- **Cyclist → person.** KITTI boxes cover rider + bike, while COCO models box only the rider. Expect low IoU, and person AP75 hurt more than AP50.
- **Dropped classes.** Tram/Misc/DontCare regions are not in the GT, so detections there count as false positives.
- **Scale.** KITTI is 1242×375; at imgsz 640 everything shrinks ~2×, so many pedestrians become COCO-"small". Compare AP_S vs AP_L.
- **Area definition.** KITTI "area" is box area; COCO uses mask area, so S/M/L buckets differ slightly.
- **GPU type.** Run everything on one GPU type, or latency numbers aren't comparable. Uncomment `--nodelist` in the sbatch to pin it. Avoid `condo` (preemptible).

## Suggested report outline (15-slide deck is the least work)

1. Title / goal
2. Models and families (YOLO11 anchor-free CNN + NMS vs RT-DETR hybrid encoder + query decoder, NMS-free)
3. Datasets + taxonomy mapping table
4. Setup: hardware, versions, thresholds, exact commands (copy from `summary.md` §5)
5–6. COCO results (Table 1)
7–8. COCO→KITTI domain shift: overall + per-class (Tables 2–3). Which class/scale drops most?
9. Speed–accuracy plot + latency breakdown (Table 4: NMS cost in postprocess for YOLO vs RT-DETR)
10. Architecture comparison: why RT-DETR postprocess is cheap (no NMS) and how it compares at similar latency
11–12. Failure analysis from `vis/` images (missed small pedestrians, cyclist box mismatch, vans→truck, FPs on trams)
13. Conclusions
14. References (≥8, ≥4 papers: YOLO/YOLOv1, RT-DETR, DETR, COCO, KITTI, FPN or Focal Loss; ≥2 docs: Ultralytics docs, pycocotools/COCO API)
15. Appendix: link to repo with `code/`, `logs/*.log`, `results/*/metrics.json`

Submit the code folder + `logs/` + `results/*/metrics.json` as the command log.
