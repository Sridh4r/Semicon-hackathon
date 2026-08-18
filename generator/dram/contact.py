"""Contact/via generation helpers."""

import numpy as np


def draw_contacts(
	canvas: np.ndarray,
	x_positions: list[int],
	y_positions: list[int],
	size: int,
	intensity: int,
) -> None:
	"""Draw square contacts at every wordline/bitline crossing."""
	half = size // 2
	height, width = canvas.shape
	for y in y_positions:
		y0 = max(0, y - half)
		y1 = min(height, y + half + (size % 2))
		for x in x_positions:
			x0 = max(0, x - half)
			x1 = min(width, x + half + (size % 2))
			canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], intensity)
