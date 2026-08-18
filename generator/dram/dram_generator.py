"""Top-level DRAM dataset generator."""

import json
import random
from pathlib import Path

from PIL import Image

from .array import build_periodic_array
from .cell_model import DRAMConfig


class DRAMGenerator:
	"""Generate DRAM-like synthetic images for MVP localization experiments."""

	def __init__(self, config: DRAMConfig) -> None:
		self.config = config
		self.config.validate()

	def _random_center(self, target_size: int, seed: int | None = None) -> tuple[int, int]:
		"""Pick a random target center that allows a full target crop."""
		if target_size <= 0:
			raise ValueError("target_size must be positive.")
		if target_size % 2 == 0:
			raise ValueError("target_size must be odd so target_x/target_y are true center pixels.")

		half = target_size // 2
		max_x = self.config.image_width - (half + 1)
		max_y = self.config.image_height - (half + 1)
		min_x = half
		min_y = half

		if min_x > max_x or min_y > max_y:
			raise ValueError("target_size is too large for current image size.")

		rng = random.Random(seed)
		target_x = rng.randint(min_x, max_x)
		target_y = rng.randint(min_y, max_y)
		return target_x, target_y

	def build_clean_image(self):
		"""Build and return the clean periodic DRAM image array."""
		return build_periodic_array(self.config)

	def choose_target_center(self, target_size: int, seed: int | None = None) -> tuple[int, int]:
		"""Public wrapper for deterministic target center selection."""
		return self._random_center(target_size=target_size, seed=seed)

	def generate_search_image(self, output_path: str | Path) -> Path:
		"""Render and save the simplified periodic DRAM search image."""
		image = build_periodic_array(self.config)
		output = Path(output_path)
		output.parent.mkdir(parents=True, exist_ok=True)
		Image.fromarray(image, mode="L").save(output)
		return output

	def generate_search_reference_and_target(
		self,
		search_output_path: str | Path,
		reference_output_path: str | Path,
		target_json_path: str | Path,
		target_size: int,
		reference_downscale_factor: float = 10.0,
		seed: int | None = None,
	) -> dict[str, int | float | str]:
		"""Generate search image, cropped target, and downscaled reference metadata."""
		if reference_downscale_factor <= 0:
			raise ValueError("reference_downscale_factor must be positive.")

		image = build_periodic_array(self.config)
		target_x, target_y = self._random_center(target_size=target_size, seed=seed)

		half = target_size // 2
		x0 = target_x - half
		y0 = target_y - half
		x1 = x0 + target_size
		y1 = y0 + target_size
		target_crop = image[y0:y1, x0:x1]

		reference_size = max(1, int(round(target_size / reference_downscale_factor)))
		reference_image = Image.fromarray(target_crop, mode="L").resize(
			(reference_size, reference_size),
			resample=Image.Resampling.BILINEAR,
		)

		search_output = Path(search_output_path)
		reference_output = Path(reference_output_path)
		target_json_output = Path(target_json_path)

		search_output.parent.mkdir(parents=True, exist_ok=True)
		reference_output.parent.mkdir(parents=True, exist_ok=True)
		target_json_output.parent.mkdir(parents=True, exist_ok=True)

		Image.fromarray(image, mode="L").save(search_output)
		reference_image.save(reference_output)

		target = {
			"target_x": target_x,
			"target_y": target_y,
			"target_size": target_size,
			"reference_size": reference_size,
			"reference_downscale_factor": reference_downscale_factor,
		}
		if seed is not None:
			target["seed"] = seed

		target_json_output.write_text(
			json.dumps(target, indent=2),
			encoding="utf-8",
		)
		return target
