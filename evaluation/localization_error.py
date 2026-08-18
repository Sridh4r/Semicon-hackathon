"""Automatic localization error evaluation and benchmark CSV export."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matching.template.template_matcher import match


def localization_error(
    ground_truth_x: int,
    ground_truth_y: int,
    predicted_x: int,
    predicted_y: int,
) -> float:
    """Compute Euclidean localization error in pixels."""
    dx = predicted_x - ground_truth_x
    dy = predicted_y - ground_truth_y
    return math.hypot(dx, dy)


def _load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def _discover_sample_bases(dataset_dir: Path) -> list[str]:
    bases: list[str] = []
    for search_path in dataset_dir.glob("sample_*_search.png"):
        name = search_path.name
        if "_preprocessed" in name:
            continue
        base = name[:-len("_search.png")]
        reference = dataset_dir / f"{base}_reference.png"
        gt_json = dataset_dir / f"{base}.json"
        if reference.exists() and gt_json.exists():
            bases.append(base)
    return sorted(set(bases))


def _sample_id_from_base(sample_base: str) -> str:
    if sample_base.startswith("sample_"):
        return sample_base[len("sample_"):]
    return sample_base


def evaluate_sample(dataset_dir: Path, sample_base: str, reference_scale: float) -> dict[str, float | int | str]:
    """Run matching and compute localization error for one sample base name."""
    search_path = dataset_dir / f"{sample_base}_search.png"
    reference_path = dataset_dir / f"{sample_base}_reference.png"
    gt_json_path = dataset_dir / f"{sample_base}.json"

    search = _load_gray(search_path)
    reference = _load_gray(reference_path)
    gt = json.loads(gt_json_path.read_text(encoding="utf-8"))

    gt_x = int(gt["target_x"])
    gt_y = int(gt["target_y"])

    t0 = time.perf_counter()
    result = match(search, reference, reference_scale=reference_scale)
    runtime_sec = time.perf_counter() - t0
    error_px = localization_error(gt_x, gt_y, result.center_x, result.center_y)

    return {
        "sample_id": _sample_id_from_base(sample_base),
        "gt_x": gt_x,
        "gt_y": gt_y,
        "pred_x": result.center_x,
        "pred_y": result.center_y,
        "error": error_px,
        "score": result.max_score,
        "runtime": runtime_sec,
    }


def write_benchmark_csv(rows: list[dict[str, float | int | str]], output_csv: Path) -> None:
    """Write benchmark rows to a CSV file."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "sample_id",
        "gt_x",
        "gt_y",
        "pred_x",
        "pred_y",
        "error",
        "score",
        "runtime",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_benchmark(dataset_dir: Path, output_csv: Path, reference_scale: float = 1.0) -> list[dict[str, float | int | str]]:
    """Evaluate all discoverable samples and export benchmark CSV."""
    sample_bases = _discover_sample_bases(dataset_dir)
    if not sample_bases:
        raise FileNotFoundError(f"No samples found in {dataset_dir}")

    rows = [evaluate_sample(dataset_dir, base, reference_scale) for base in sample_bases]
    write_benchmark_csv(rows, output_csv)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute localization error benchmark CSV")
    parser.add_argument("--dataset-dir", default=".")
    parser.add_argument("--output-csv", default="benchmark_localization.csv")
    parser.add_argument("--reference-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    output_csv = Path(args.output_csv).resolve()

    rows = run_benchmark(dataset_dir, output_csv, reference_scale=args.reference_scale)

    print(f"Evaluated samples: {len(rows)}")
    print(f"Benchmark CSV: {output_csv}")

    if rows:
        avg_error = sum(float(row["error"]) for row in rows) / len(rows)
        print(f"Average error (px): {avg_error:.4f}")
        print("First row:")
        print(json.dumps(rows[0], indent=2))


if __name__ == "__main__":
    main()
