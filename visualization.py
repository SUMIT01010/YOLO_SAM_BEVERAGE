"""Visualization helpers for YOLO + SAM pipeline outputs."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np

from pipeline_infer import InferenceResult


def _bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _render_mask(mask: np.ndarray) -> np.ndarray:
    return (mask.astype(np.uint8) * 255)


def save_example_panel(result: InferenceResult, output_path: str | Path) -> None:
    """Save one side-by-side panel for a single image result."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"{Path(result.image_path).name} | status={result.status}", fontsize=10)

    # 1) Original
    axes[0].set_title("Input")
    if result.original_bgr is not None:
        axes[0].imshow(_bgr_to_rgb(result.original_bgr))
    else:
        axes[0].text(0.5, 0.5, "Image load failed", ha="center", va="center")
    axes[0].axis("off")

    # 2) YOLO detections
    axes[1].set_title("YOLO Detection")
    if result.yolo_overlay_bgr is not None:
        axes[1].imshow(_bgr_to_rgb(result.yolo_overlay_bgr))
    else:
        axes[1].text(0.5, 0.5, "No detections", ha="center", va="center")
    axes[1].axis("off")

    # 3) SAM mask
    axes[2].set_title("SAM Mask")
    if result.mask is not None:
        axes[2].imshow(_render_mask(result.mask), cmap="gray")
    else:
        axes[2].text(0.5, 0.5, "No mask", ha="center", va="center")
    axes[2].axis("off")

    # 4) Final clean extraction
    axes[3].set_title("Extracted Bottle")
    if result.extracted_bgr is not None:
        axes[3].imshow(_bgr_to_rgb(result.extracted_bgr))
    else:
        axes[3].text(0.5, 0.5, "No extraction", ha="center", va="center")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_example_panels(
    results: Sequence[InferenceResult],
    output_dir: str | Path,
    max_examples: int = 10,
) -> List[Path]:
    """Save N example panels and return output paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    for i, result in enumerate(list(results)[:max_examples], start=1):
        panel_path = output_dir / f"example_{i:02d}.png"
        save_example_panel(result, panel_path)
        saved_paths.append(panel_path)
    return saved_paths


def save_examples_grid(
    results: Sequence[InferenceResult],
    output_path: str | Path,
    max_examples: int = 10,
) -> Path:
    """Save one grid with N rows x 4 columns (input, yolo, mask, extracted)."""
    selected = list(results)[:max_examples]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not selected:
        raise ValueError("No results available to visualize.")

    rows = len(selected)
    fig, axes = plt.subplots(rows, 4, figsize=(16, max(3 * rows, 4)))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, result in enumerate(selected):
        row_axes = axes[row_idx]

        # Input
        row_axes[0].set_title("Input")
        if result.original_bgr is not None:
            row_axes[0].imshow(_bgr_to_rgb(result.original_bgr))
        else:
            row_axes[0].text(0.5, 0.5, "Load failed", ha="center", va="center")
        row_axes[0].axis("off")

        # YOLO
        row_axes[1].set_title("YOLO")
        if result.yolo_overlay_bgr is not None:
            row_axes[1].imshow(_bgr_to_rgb(result.yolo_overlay_bgr))
        else:
            row_axes[1].text(0.5, 0.5, "No detection", ha="center", va="center")
        row_axes[1].axis("off")

        # SAM
        row_axes[2].set_title("SAM Mask")
        if result.mask is not None:
            row_axes[2].imshow(_render_mask(result.mask), cmap="gray")
        else:
            row_axes[2].text(0.5, 0.5, "No mask", ha="center", va="center")
        row_axes[2].axis("off")

        # Extracted
        row_axes[3].set_title("Extracted")
        if result.extracted_bgr is not None:
            row_axes[3].imshow(_bgr_to_rgb(result.extracted_bgr))
        else:
            row_axes[3].text(0.5, 0.5, "No extraction", ha="center", va="center")
        row_axes[3].axis("off")

        row_axes[0].set_ylabel(
            f"{row_idx + 1:02d}\n{Path(result.image_path).name}\n{result.status}",
            rotation=0,
            labelpad=40,
            fontsize=8,
            va="center",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_clean_crops(
    results: Sequence[InferenceResult],
    output_dir: str | Path,
    max_examples: int = 10,
) -> List[Path]:
    """Save extracted clean bottle crops as image files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for i, result in enumerate(list(results)[:max_examples], start=1):
        if result.extracted_bgr is None:
            continue
        out_path = output_dir / f"clean_{i:02d}_{Path(result.image_path).stem}.png"
        cv2.imwrite(str(out_path), result.extracted_bgr)
        saved.append(out_path)
    return saved

