"""Dataset preparation utilities for YOLOv8 training.

This module:
1) Reads CSV annotations.
2) Selects first N images from the image folder.
3) Creates stratified train/val/test splits.
4) Converts bounding boxes to YOLO text labels (single class: bottle).
5) Writes dataset folders + data.yaml.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DETECTION_CLASS_NAME = "bottle"


def load_annotations(csv_path: Path) -> pd.DataFrame:
    """Load annotations CSV and validate required columns."""
    required_columns = {
        "filename",
        "width",
        "height",
        "class",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    }
    df = pd.read_csv(csv_path)
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required annotation columns: {sorted(missing)}")

    df = df[list(required_columns)].copy()
    for col in ["width", "height", "xmin", "ymin", "xmax", "ymax"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    df["class"] = df["class"].astype(str)
    df["filename"] = df["filename"].astype(str)
    return df


def list_first_n_annotated_images(
    image_dir: Path,
    annotated_filenames: Sequence[str],
    max_images: int,
) -> List[str]:
    """Return first N image filenames (sorted) that have annotations."""
    annotated_set = set(annotated_filenames)
    image_files = sorted(
        p.name
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    selected = [name for name in image_files if name in annotated_set][:max_images]

    if len(selected) < max_images:
        raise ValueError(
            f"Only {len(selected)} annotated images found, but requested {max_images}."
        )
    return selected


def build_image_profiles(df: pd.DataFrame, filenames: Sequence[str]) -> pd.DataFrame:
    """Build per-image class profile used for stratified splitting."""
    filtered = df[df["filename"].isin(filenames)].copy()

    grouped = filtered.groupby("filename")["class"].apply(
        lambda s: sorted(set(s.tolist()))
    )
    profile = grouped.reset_index(name="classes")
    profile["label_key"] = profile["classes"].apply(lambda x: "|".join(x))
    profile["primary_class"] = profile["classes"].apply(lambda x: x[0])
    return profile


def stratified_split_filenames(
    profile_df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> Dict[str, List[str]]:
    """Split images into train/val/test using stratification.

    Primary strategy: stratify by full image class profile.
    Fallback strategy: stratify by primary class.
    Last fallback: random split (if class profile is too sparse).
    """
    if not abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    filenames = profile_df["filename"].tolist()
    n_total = len(filenames)
    if n_total == 0:
        raise ValueError("No images available for splitting.")

    n_test = int(round(n_total * test_ratio))
    n_val = int(round(n_total * val_ratio))
    n_train = n_total - n_test - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(
            f"Invalid split sizes: train={n_train}, val={n_val}, test={n_test}"
        )

    # Strategy 1: full label profile stratification.
    stratify_labels = profile_df["label_key"]
    try:
        train_val_files, test_files = train_test_split(
            filenames,
            test_size=n_test,
            random_state=random_state,
            stratify=stratify_labels,
        )

        train_val_df = profile_df.set_index("filename").loc[train_val_files].reset_index()
        train_files, val_files = train_test_split(
            train_val_files,
            test_size=n_val,
            random_state=random_state,
            stratify=train_val_df["label_key"],
        )
        return {"train": sorted(train_files), "val": sorted(val_files), "test": sorted(test_files)}
    except ValueError:
        pass

    # Strategy 2: primary-class stratification.
    try:
        train_val_files, test_files = train_test_split(
            filenames,
            test_size=n_test,
            random_state=random_state,
            stratify=profile_df["primary_class"],
        )
        train_val_df = profile_df.set_index("filename").loc[train_val_files].reset_index()
        train_files, val_files = train_test_split(
            train_val_files,
            test_size=n_val,
            random_state=random_state,
            stratify=train_val_df["primary_class"],
        )
        return {"train": sorted(train_files), "val": sorted(val_files), "test": sorted(test_files)}
    except ValueError:
        pass

    # Strategy 3: unstratified random split.
    train_val_files, test_files = train_test_split(
        filenames,
        test_size=n_test,
        random_state=random_state,
        shuffle=True,
    )
    train_files, val_files = train_test_split(
        train_val_files,
        test_size=n_val,
        random_state=random_state,
        shuffle=True,
    )
    return {"train": sorted(train_files), "val": sorted(val_files), "test": sorted(test_files)}


def build_class_mapping(df: pd.DataFrame, class_order: Sequence[str] | None = None) -> Dict[str, int]:
    """Build class name -> id mapping."""
    if class_order is None:
        classes = sorted(df["class"].unique().tolist())
    else:
        classes = list(class_order)
    return {cls_name: idx for idx, cls_name in enumerate(classes)}


def _clamp_bbox(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    width: float,
    height: float,
) -> Tuple[float, float, float, float]:
    xmin = max(0.0, min(xmin, width - 1))
    ymin = max(0.0, min(ymin, height - 1))
    xmax = max(0.0, min(xmax, width))
    ymax = max(0.0, min(ymax, height))
    if xmax <= xmin:
        xmax = min(width, xmin + 1.0)
    if ymax <= ymin:
        ymax = min(height, ymin + 1.0)
    return xmin, ymin, xmax, ymax


def row_to_yolo_line(row: pd.Series, _class_to_id: Dict[str, int]) -> str:
    """Convert one CSV annotation row to YOLO label format."""
    width = float(row["width"])
    height = float(row["height"])
    xmin, ymin, xmax, ymax = _clamp_bbox(
        float(row["xmin"]),
        float(row["ymin"]),
        float(row["xmax"]),
        float(row["ymax"]),
        width,
        height,
    )

    x_center = ((xmin + xmax) / 2.0) / width
    y_center = ((ymin + ymax) / 2.0) / height
    box_w = (xmax - xmin) / width
    box_h = (ymax - ymin) / height

    # Detection-only dataset: every annotation is class 0 ("bottle").
    class_id = 0
    return f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"


def write_split_data(
    df: pd.DataFrame,
    split_name: str,
    split_filenames: Sequence[str],
    source_image_dir: Path,
    dataset_root: Path,
    class_to_id: Dict[str, int],
) -> None:
    """Copy images and write YOLO label files for a split."""
    images_out = dataset_root / "images" / split_name
    labels_out = dataset_root / "labels" / split_name
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    grouped = {k: g for k, g in df.groupby("filename")}
    for filename in split_filenames:
        src_image = source_image_dir / filename
        if not src_image.exists():
            raise FileNotFoundError(f"Image not found: {src_image}")

        shutil.copy2(src_image, images_out / filename)
        label_path = labels_out / f"{Path(filename).stem}.txt"

        rows = grouped.get(filename)
        if rows is None:
            # No annotations -> empty label file for YOLO compatibility.
            label_path.write_text("", encoding="utf-8")
            continue

        yolo_lines = [row_to_yolo_line(row, class_to_id) for _, row in rows.iterrows()]
        label_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")


def write_data_yaml(dataset_root: Path, class_to_id: Dict[str, int]) -> Path:
    """Write YOLO data.yaml file and return its path."""
    expected_mapping = {DETECTION_CLASS_NAME: 0}
    if class_to_id != expected_mapping:
        raise ValueError(
            f"Expected single-class mapping {expected_mapping}, got {class_to_id}"
        )

    lines = [
        f"path: {dataset_root.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "nc: 1",
        'names: ["bottle"]',
    ]

    yaml_path = dataset_root / "data.yaml"
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def summarize_splits(
    df: pd.DataFrame,
    splits: Dict[str, Sequence[str]],
    class_names: Sequence[str],
) -> Dict[str, Dict[str, object]]:
    """Build split summary to verify distribution quality."""
    summary: Dict[str, Dict[str, object]] = {}

    for split_name, filenames in splits.items():
        split_df = df[df["filename"].isin(filenames)].copy()
        bbox_counts = {cls: int((split_df["class"] == cls).sum()) for cls in class_names}

        image_presence = {}
        for cls in class_names:
            cls_files = split_df.loc[split_df["class"] == cls, "filename"].nunique()
            image_presence[cls] = int(cls_files)

        summary[split_name] = {
            "num_images": int(len(filenames)),
            "num_boxes": int(len(split_df)),
            "bbox_per_class": bbox_counts,
            "images_containing_class": image_presence,
        }

    return summary


def prepare_yolo_dataset(
    annotations_csv: str | Path,
    image_dir: str | Path,
    output_dir: str | Path,
    max_images: int = 1000,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    random_state: int = 42,
    class_order: Sequence[str] | None = None,
) -> Dict[str, object]:
    """Prepare YOLO-ready dataset and return metadata."""
    annotations_csv = Path(annotations_csv)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_annotations(annotations_csv)
    selected_files = list_first_n_annotated_images(
        image_dir=image_dir,
        annotated_filenames=df["filename"].unique().tolist(),
        max_images=max_images,
    )
    selected_df = df[df["filename"].isin(selected_files)].copy()
    # Detection-only training: original class labels are intentionally ignored.
    selected_df["class"] = DETECTION_CLASS_NAME

    profile_df = build_image_profiles(selected_df, selected_files)
    splits = stratified_split_filenames(
        profile_df=profile_df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
    )

    _ = class_order  # Preserved for backward compatibility; ignored in single-class mode.
    class_to_id = {DETECTION_CLASS_NAME: 0}
    class_names = [DETECTION_CLASS_NAME]

    for split_name, filenames in splits.items():
        write_split_data(
            df=selected_df,
            split_name=split_name,
            split_filenames=filenames,
            source_image_dir=image_dir,
            dataset_root=output_dir,
            class_to_id=class_to_id,
        )

    data_yaml = write_data_yaml(output_dir, class_to_id)
    split_summary = summarize_splits(selected_df, splits, class_names)

    manifest_rows = []
    for split_name, filenames in splits.items():
        for fname in filenames:
            manifest_rows.append({"filename": fname, "split": split_name})
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "split_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    summary_path = output_dir / "split_summary.json"
    summary_path.write_text(json.dumps(split_summary, indent=2), encoding="utf-8")

    return {
        "data_yaml": str(data_yaml),
        "manifest_csv": str(manifest_path),
        "summary_json": str(summary_path),
        "class_to_id": class_to_id,
        "class_names": class_names,
        "splits": {k: list(v) for k, v in splits.items()},
    }
