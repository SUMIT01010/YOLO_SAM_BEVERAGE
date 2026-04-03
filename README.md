# Beverage Bottle Detection + SAM Segmentation Pipeline (Experiment)

This document defines the full process for your dataset in:

`/Users/sumitsaurabh/Desktop/DSL/SPRITE_COLA_FANTA`

Goal:
- Train YOLOv8 detector from CSV bounding boxes.
- Detect bottle(s) in noisy image.
- Use SAM with YOLO box prompt to segment the bottle.
- Remove background with mask.
- Produce a clean extracted bottle image.

## 1) Dataset Input

Annotation CSV:
- `SPRITE_COLA_FANTA/_annotations.csv`

CSV schema:
- `filename,width,height,class,xmin,ymin,xmax,ymax`

Detected classes in CSV:
- `coca-cola`
- `fanta`
- `sprite`

## 2) Data Preparation Plan (First 250 Images)

We will use only the first `250` selected images for the initial experiment.

Split target:
- Train: `70%` -> `175` images
- Validation: `10%` -> `25` images
- Test: `20%` -> `50` images

Stratification requirement:
- Keep class distribution balanced across train/val/test.
- Because images can contain multiple bottles/classes, stratification should be done on image-level class profile (multi-label aware), then validated with per-class box counts.

Output structure (YOLO format):
- `datasets/beverage_yolo/images/train`
- `datasets/beverage_yolo/images/val`
- `datasets/beverage_yolo/images/test`
- `datasets/beverage_yolo/labels/train`
- `datasets/beverage_yolo/labels/val`
- `datasets/beverage_yolo/labels/test`
- `datasets/beverage_yolo/data.yaml`

## 3) CSV -> YOLO Label Conversion

For each annotation row:
- Read image width/height.
- Convert box to YOLO normalized format:
  - `x_center = ((xmin + xmax) / 2) / width`
  - `y_center = ((ymin + ymax) / 2) / height`
  - `w = (xmax - xmin) / width`
  - `h = (ymax - ymin) / height`
- Write one line per box:
  - `<class_id> <x_center> <y_center> <w> <h>`

Class-to-id map:
- `coca-cola: 0`
- `fanta: 1`
- `sprite: 2`

## 4) YOLOv8 Training

Model:
- `ultralytics` YOLOv8 (`yolov8n.pt` or `yolov8s.pt` as starting checkpoint)

Training config (experiment run):
- `imgsz=640`
- `epochs=25`
- `batch=16` (adjust for GPU memory)
- `data=datasets/beverage_yolo/data.yaml`

Expected artifact:
- `runs/detect/train*/weights/best.pt`

## 5) Inference Pipeline (YOLO + SAM)

For a new noisy input image:

1. YOLO detect bottle bounding boxes.
2. Select target box (highest confidence or top-k loop).
3. Pass box as prompt to SAM predictor.
4. Get binary mask from SAM output.
5. Clean background:
   - Keep masked object pixels.
   - Set background to black (or transparent RGBA).
   - Crop tightly around mask region.
6. Save final clean bottle image.

## 6) Visualization Output

For final visual check, save side-by-side results with:
- Original noisy image
- YOLO bounding box overlay
- SAM binary mask
- Background-removed bottle crop

Suggested output path:
- `outputs/pipeline_visualization.png`

## 7) Show 10 Example Results

At the end of this experiment, generate `10` examples from the test split and save:
- Per-image visualization files:
  - `outputs/examples/example_01.png` ... `example_10.png`
- One combined grid:
  - `outputs/examples/pipeline_examples_grid.png`

Each example should display:
- Input image
- YOLO detection result
- SAM mask
- Final clear extracted bottle

## 8) Clean Project Structure (Required Flow)

The flow should be:
- `main.py` contains the major `main()` function and acts as the single entrypoint.
- All other logic should be implemented in separate `.py` files as reusable functions.

Recommended layout:

```text
DSL/
├── main.py                      # Major orchestrator function (entrypoint)
├── prepare_dataset.py           # CSV parsing, 250-image selection, stratified split, YOLO labels
├── train_yolo.py                # YOLOv8 training wrapper
├── pipeline_infer.py            # YOLO + SAM inference and background cleanup
├── visualization.py             # Save 10 examples + final grid
├── read.md
└── SPRITE_COLA_FANTA/
```

Module responsibility:

- `prepare_dataset.py`
  - Load CSV
  - Select 250 images
  - Stratified 70/10/20 split
  - Convert labels to YOLO format
  - Write `data.yaml`

- `train_yolo.py`
  - Train detector via ultralytics API

- `pipeline_infer.py`
  - Load YOLO model
  - Load SAM model
  - Run YOLO + SAM end-to-end inference
  - Save clean extracted bottle image
 
- `visualization.py`
  - Save per-image visualizations for 10 samples
  - Save combined grid image

- `main.py`
  - Required single CLI entrypoint
  - Contains the major `main()` function
  - Calls:
    - dataset preparation
    - YOLO training (`epochs=25`)
    - YOLO+SAM inference
    - 10-example visualization generation

## 9) Required Dependencies

Install (example):

```bash
pip install ultralytics segment-anything opencv-python pillow matplotlib pandas scikit-learn torch
```

Notes:
- SAM also needs a checkpoint file, for example `sam_vit_h_4b8939.pth`.

## 10) Constraints Confirmed

- EfficientNet is not included in this experiment.
- YOLOv8 is used for detection.
- SAM is used for segmentation from YOLO box prompt.
- PyTorch is used for model inference flow.
- Pipeline will be modular with clean function separation.

## 11) Next Step

After this `read.md`, implementation will proceed in code with:
- Dataset conversion + stratified split (first 250 images)
- YOLO training script
- YOLO + SAM inference script for clean bottle extraction
- Visualization generation for 10 example images from this pipeline
- `main.py` as the top-level orchestrator for the full flow
