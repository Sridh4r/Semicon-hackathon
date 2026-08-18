"""Wordline synthesis utilities."""

import numpy as np


def draw_wordlines(
	canvas: np.ndarray,
	y_positions: list[int],
	thickness: int,
	intensity: int,
) -> None:
	"""Draw horizontal wordlines across the canvas."""
	half = thickness // 2
	height = canvas.shape[0]
	for y in y_positions:
		y0 = max(0, y - half)
		y1 = min(height, y + half + (thickness % 2))
		canvas[y0:y1, :] = np.maximum(canvas[y0:y1, :], intensity)
