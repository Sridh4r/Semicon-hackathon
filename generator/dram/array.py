"""Array assembly logic for DRAM cells."""

import numpy as np

from .bitline import draw_bitlines
from .cell_model import DRAMConfig
from .contact import draw_contacts
from .wordline import draw_wordlines


def build_periodic_array(config: DRAMConfig) -> np.ndarray:
	"""Build a simplified periodic DRAM array image."""
	config.validate()
	image = np.full(
		(config.image_height, config.image_width),
		fill_value=config.background_intensity,
		dtype=np.uint8,
	)

	x_positions = config.x_positions()
	y_positions = config.y_positions()

	draw_wordlines(
		canvas=image,
		y_positions=y_positions,
		thickness=config.wordline_thickness,
		intensity=config.wordline_intensity,
	)
	draw_bitlines(
		canvas=image,
		x_positions=x_positions,
		thickness=config.bitline_thickness,
		intensity=config.bitline_intensity,
	)
	draw_contacts(
		canvas=image,
		x_positions=x_positions,
		y_positions=y_positions,
		size=config.contact_size,
		intensity=config.contact_intensity,
	)

	return image
