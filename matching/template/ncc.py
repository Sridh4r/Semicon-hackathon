"""Normalized cross-correlation (NCC) matcher."""

from __future__ import annotations

import numpy as np
import cv2


def compute_ncc_score_map(search_image: np.ndarray, reference_image: np.ndarray) -> np.ndarray:
	"""Compute normalized correlation score map for template matching.

	MVP baseline uses TM_CCOEFF_NORMED as requested.
	"""
	search = search_image.astype(np.float32)
	reference = reference_image.astype(np.float32)
	return cv2.matchTemplate(search, reference, cv2.TM_CCOEFF_NORMED)
