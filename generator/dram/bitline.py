"""Bitline synthesis utilities."""

import numpy as np


def draw_bitlines(
	canvas: np.ndarray,
	x_positions: list[int],
	thickness: int,
	intensity: int,
) -> None:
	"""Draw vertical bitlines across the canvas."""
	half = thickness // 2
	width = canvas.shape[1]
	for x in x_positions:
		x0 = max(0, x - half)
		x1 = min(width, x + half + (thickness % 2))
		canvas[:, x0:x1] = np.maximum(canvas[:, x0:x1], intensity)
