"""Template matching orchestrator."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .zncc import compute_zncc_score_map


@dataclass(frozen=True)
class MatchResult:
	"""Result of template matching."""

	score_map: np.ndarray
	max_score: float
	top_left_x: int
	top_left_y: int
	center_x: int
	center_y: int
	matched_reference_width: int
	matched_reference_height: int


def _to_gray_u8(image: np.ndarray) -> np.ndarray:
	if image.ndim == 2:
		return image.astype(np.uint8)
	if image.ndim == 3 and image.shape[2] in (3, 4):
		return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
	raise ValueError("Unsupported image shape for matcher")


def resize_reference(reference_image: np.ndarray, scale: float) -> np.ndarray:
	"""Resize reference before matching."""
	if scale <= 0:
		raise ValueError("scale must be > 0")
	if scale == 1.0:
		return reference_image
	h, w = reference_image.shape[:2]
	new_w = max(1, int(round(w * scale)))
	new_h = max(1, int(round(h * scale)))
	return cv2.resize(reference_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def match(search_image: np.ndarray, reference_image: np.ndarray, reference_scale: float = 1.0) -> MatchResult:
	"""Run baseline template matching pipeline.

	Pipeline:
	Search + Reference -> Resize reference -> matchTemplate -> score map ->
	maximum score -> top-left coordinate -> center coordinate.
	"""
	search = _to_gray_u8(search_image)
	reference = _to_gray_u8(reference_image)
	reference = resize_reference(reference, reference_scale)

	if reference.shape[0] > search.shape[0] or reference.shape[1] > search.shape[1]:
		raise ValueError("reference must be smaller than or equal to search image")

	score_map = compute_zncc_score_map(search, reference)
	_, max_score, _, max_loc = cv2.minMaxLoc(score_map)
	top_left_x, top_left_y = int(max_loc[0]), int(max_loc[1])

	ref_h, ref_w = reference.shape[:2]
	center_x = top_left_x + (ref_w // 2)
	center_y = top_left_y + (ref_h // 2)

	return MatchResult(
		score_map=score_map,
		max_score=float(max_score),
		top_left_x=top_left_x,
		top_left_y=top_left_y,
		center_x=center_x,
		center_y=center_y,
		matched_reference_width=ref_w,
		matched_reference_height=ref_h,
	)
