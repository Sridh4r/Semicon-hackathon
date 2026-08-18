"""Zero-mean normalized cross-correlation (ZNCC) matcher."""

from __future__ import annotations

import numpy as np
import cv2


def compute_zncc_score_map(search_image: np.ndarray, reference_image: np.ndarray) -> np.ndarray:
	"""Compute ZNCC-like score map using TM_CCOEFF_NORMED.

	OpenCV TM_CCOEFF_NORMED performs normalized correlation with mean compensation,
	which is the requested baseline for this project stage.
	"""
	search = search_image.astype(np.float32)
	reference = reference_image.astype(np.float32)
	return cv2.matchTemplate(search, reference, cv2.TM_CCOEFF_NORMED)
