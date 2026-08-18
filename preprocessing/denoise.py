"""Noise reduction utilities."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def light_denoise(image: np.ndarray, radius: float = 0.6) -> np.ndarray:
	"""Apply a mild blur to suppress high-frequency noise without over-smoothing."""
	if radius < 0:
		raise ValueError("radius must be non-negative")
	if radius == 0:
		return image.copy()

	pil = Image.fromarray(image.astype(np.uint8), mode="L")
	out = pil.filter(ImageFilter.GaussianBlur(radius=radius))
	return np.asarray(out, dtype=np.uint8)
