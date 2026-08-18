"""Normalization utilities."""

from __future__ import annotations

import numpy as np


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
	"""Min-max normalize an image to uint8 [0, 255]."""
	arr = image.astype(np.float32)
	min_val = float(arr.min())
	max_val = float(arr.max())
	if max_val <= min_val:
		return np.zeros_like(arr, dtype=np.uint8)
	scaled = (arr - min_val) * (255.0 / (max_val - min_val))
	return np.clip(scaled, 0, 255).astype(np.uint8)


def normalize_percentile(
	image: np.ndarray,
	low_percentile: float = 1.0,
	high_percentile: float = 99.0,
) -> np.ndarray:
	"""Percentile-based normalization to reduce outlier influence."""
	if not 0 <= low_percentile < high_percentile <= 100:
		raise ValueError("percentile range must satisfy 0 <= low < high <= 100")

	arr = image.astype(np.float32)
	lo = float(np.percentile(arr, low_percentile))
	hi = float(np.percentile(arr, high_percentile))
	if hi <= lo:
		return np.zeros_like(arr, dtype=np.uint8)

	clipped = np.clip(arr, lo, hi)
	scaled = (clipped - lo) * (255.0 / (hi - lo))
	return np.clip(scaled, 0, 255).astype(np.uint8)
