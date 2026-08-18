"""End-to-end preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .denoise import light_denoise
from .normalize import normalize_percentile


@dataclass(frozen=True)
class ExperimentParams:
	"""Parameters controlling synthetic SEM-like degradations."""

	gaussian_sigma: float = 10.0
	blur_radius: float = 1.2
	contrast_jitter: float = 0.25


@dataclass(frozen=True)
class PreprocessingParams:
	"""Minimal preprocessing controls to preserve tiny DRAM details."""

	normalize_low_percentile: float = 1.0
	normalize_high_percentile: float = 99.0
	denoise_radius: float = 0.6


def to_grayscale(image: np.ndarray) -> np.ndarray:
	"""Convert input image to grayscale uint8."""
	if image.ndim == 2:
		return image.astype(np.uint8)
	if image.ndim == 3 and image.shape[2] in (3, 4):
		# Standard luminance conversion from RGB channels.
		rgb = image[..., :3].astype(np.float32)
		gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
		return np.clip(gray, 0, 255).astype(np.uint8)
	raise ValueError("Unsupported image shape for grayscale conversion")


def minimal_preprocess(
	image: np.ndarray,
	params: PreprocessingParams | None = None,
) -> np.ndarray:
	"""Apply only grayscale, intensity normalization, and light denoise."""
	cfg = params or PreprocessingParams()
	gray = to_grayscale(image)
	normalized = normalize_percentile(
		gray,
		low_percentile=cfg.normalize_low_percentile,
		high_percentile=cfg.normalize_high_percentile,
	)
	return light_denoise(normalized, radius=cfg.denoise_radius)


def preprocess_search_and_reference(
	search_image: np.ndarray,
	reference_image: np.ndarray,
	params: PreprocessingParams | None = None,
) -> tuple[np.ndarray, np.ndarray]:
	"""Run the same minimal preprocessing on search and reference images."""
	cfg = params or PreprocessingParams()
	return minimal_preprocess(search_image, cfg), minimal_preprocess(reference_image, cfg)


def apply_gaussian_noise(image: np.ndarray, sigma: float, seed: int) -> np.ndarray:
	"""Apply additive Gaussian noise with deterministic seed."""
	rng = np.random.default_rng(seed)
	noise = rng.normal(loc=0.0, scale=sigma, size=image.shape)
	out = image.astype(np.float32) + noise
	return np.clip(out, 0, 255).astype(np.uint8)


def apply_blur(image: np.ndarray, radius: float) -> np.ndarray:
	"""Apply Gaussian blur using Pillow."""
	pil = Image.fromarray(image, mode="L")
	blurred = pil.filter(ImageFilter.GaussianBlur(radius=radius))
	return np.asarray(blurred, dtype=np.uint8)


def apply_contrast_variation(image: np.ndarray, factor: float) -> np.ndarray:
	"""Apply deterministic contrast factor."""
	pil = Image.fromarray(image, mode="L")
	enhanced = ImageEnhance.Contrast(pil).enhance(factor)
	return np.asarray(enhanced, dtype=np.uint8)


def contrast_factor_for_sample(sample_id: int, base_seed: int | None, jitter: float) -> float:
	"""Create a deterministic contrast factor per sample."""
	effective_seed = (base_seed or 0) * 100003 + sample_id * 9973
	rng = np.random.default_rng(effective_seed)
	delta = rng.uniform(-jitter, jitter)
	return float(1.0 + delta)


def run_experiment(
	clean_image: np.ndarray,
	experiment: str,
	sample_id: int,
	noise_seed: int | None,
	base_seed: int | None,
	params: ExperimentParams,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
	"""Apply a selected experiment recipe to the clean image."""
	code = experiment.upper()
	meta: dict[str, float | int | str] = {"experiment": code}

	if code == "A":
		return clean_image.copy(), meta

	if code == "B":
		if noise_seed is None:
			raise ValueError("Experiment B requires noise_seed.")
		meta["noise_seed"] = noise_seed
		meta["gaussian_sigma"] = params.gaussian_sigma
		return apply_gaussian_noise(clean_image, sigma=params.gaussian_sigma, seed=noise_seed), meta

	if code == "C":
		meta["blur_radius"] = params.blur_radius
		return apply_blur(clean_image, radius=params.blur_radius), meta

	if code == "D":
		factor = contrast_factor_for_sample(sample_id, base_seed, params.contrast_jitter)
		meta["contrast_factor"] = factor
		return apply_contrast_variation(clean_image, factor=factor), meta

	if code == "E":
		if noise_seed is None:
			raise ValueError("Experiment E requires noise_seed.")
		factor = contrast_factor_for_sample(sample_id, base_seed, params.contrast_jitter)
		out = apply_gaussian_noise(clean_image, sigma=params.gaussian_sigma, seed=noise_seed)
		out = apply_blur(out, radius=params.blur_radius)
		out = apply_contrast_variation(out, factor=factor)
		meta["noise_seed"] = noise_seed
		meta["gaussian_sigma"] = params.gaussian_sigma
		meta["blur_radius"] = params.blur_radius
		meta["contrast_factor"] = factor
		return out, meta

	raise ValueError(f"Unsupported experiment code: {experiment}")
