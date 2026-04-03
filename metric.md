# Project Metrics

Last updated: 2026-03-28

## Scope

- Pipeline: YOLOv8 detection + MobileSAM segmentation + clean crop export
- Primary run: `runs/detect/runs/detect/beverage_exp25`
- Sources:
  - `datasets/beverage_yolo/split_summary.json`
  - `runs/detect/runs/detect/beverage_exp25/results.csv`
  - `outputs/run_summary.json`

## 1) Dataset Metrics

| Split | Images | Boxes | Avg boxes/image |
|---|---:|---:|---:|
| Train | 700 | 2170 | 3.10 |
| Val | 100 | 281 | 2.81 |
| Test | 200 | 549 | 2.75 |
| Total | 1000 | 3000 | 3.00 |

- Class setup in `data.yaml`: single class (`bottle`, `nc: 1`)

## 2) YOLO Training Metrics (`beverage_exp25`)

### Best epoch metrics

- Best `mAP50`: `0.80306` (epoch `9`)
- Best `mAP50-95`: `0.57502` (epoch `9`)
- Best precision: `0.90455` (epoch `9`)
- Best recall: `0.72222` (epoch `9`)

### Final epoch metrics

- Final epoch: `12`
- Precision: `0.83369`
- Recall: `0.42444`
- `mAP50`: `0.43636`
- `mAP50-95`: `0.31193`

### Runtime

- Logged training wall time: `986.95s` (~`16m 27s`)

## 3) Inference/Artifact Metrics

- Status counts from `outputs/run_summary.json`:
  - `ok: 10`
- Example panels generated: `10` (`outputs/examples/example_01.png` ... `example_10.png`)
- Clean crops currently present: `19` (`outputs/clean_crops/`)
- Combined grid: `outputs/examples/pipeline_examples_grid.png`
- Single preview: `outputs/pipeline_visualization.png`

## 4) Metric Definitions

- Precision: fraction of predicted boxes that are correct.
- Recall: fraction of ground-truth boxes that are found.
- mAP50: mean average precision at IoU 0.50.
- mAP50-95: mean AP averaged across IoU thresholds 0.50 to 0.95.

## 5) Refresh Commands

```bash
# Recompute best/final metrics from YOLO results
awk -F',' 'NR==1{for(i=1;i<=NF;i++){if($i=="metrics/precision(B)")p=i; if($i=="metrics/recall(B)")r=i; if($i=="metrics/mAP50(B)")m50=i; if($i=="metrics/mAP50-95(B)")m95=i; if($i=="epoch")e=i;} next} {if($m50+0>best_m50){best_m50=$m50+0; best_m50_epoch=$e+0;} if($m95+0>best_m95){best_m95=$m95+0; best_m95_epoch=$e+0;} if($p+0>best_p){best_p=$p+0; best_p_epoch=$e+0;} if($r+0>best_r){best_r=$r+0; best_r_epoch=$e+0;} last_e=$e+0; last_p=$p+0; last_r=$r+0; last_m50=$m50+0; last_m95=$m95+0} END{printf("best_mAP50=%.5f (epoch %d)\nbest_mAP50_95=%.5f (epoch %d)\nbest_precision=%.5f (epoch %d)\nbest_recall=%.5f (epoch %d)\nfinal_epoch=%d\nfinal_precision=%.5f\nfinal_recall=%.5f\nfinal_mAP50=%.5f\nfinal_mAP50_95=%.5f\n",best_m50,best_m50_epoch,best_m95,best_m95_epoch,best_p,best_p_epoch,best_r,best_r_epoch,last_e,last_p,last_r,last_m50,last_m95)}' runs/detect/runs/detect/beverage_exp25/results.csv
```
