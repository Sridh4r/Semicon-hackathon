"""Gradient extraction helpers."""

from __future__ import annotations

import numpy as np


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
	"""Return simple gradient magnitude image (uint8) for debug visualization."""
	arr = image.astype(np.float32)
	gy, gx = np.gradient(arr)
	mag = np.sqrt(gx * gx + gy * gy)
	if float(mag.max()) == 0.0:
		return np.zeros_like(arr, dtype=np.uint8)
	mag = (mag / float(mag.max())) * 255.0
	return np.clip(mag, 0, 255).astype(np.uint8)
