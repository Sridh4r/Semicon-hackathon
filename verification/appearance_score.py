"""Appearance scoring using ZNCC template matching."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import cv2


def match_template_local(
    search_image: np.ndarray,
    reference_image: np.ndarray,
    reference_scale: float = 1.0,
):
    """Local template matching using ZNCC."""
    from matching.template.template_matcher import match
    return match(search_image, reference_image, reference_scale=reference_scale)


def compute_appearance_score(
    search_image: np.ndarray,
    reference_image: np.ndarray,
    x: float,
    y: float,
    reference_scale: float = 1.0,
) -> float:
    """Compute appearance similarity score using local template matching.
    
    Args:
        search_image: Search image
        reference_image: Reference template
        x: X coordinate of candidate position
        y: Y coordinate of candidate position
        reference_scale: Scale factor for reference
        
    Returns:
        Appearance similarity score in [0, 1]
    """
    # Convert to grayscale if needed
    if search_image.ndim == 3:
        if search_image.shape[2] == 3:
            search_image = cv2.cvtColor(search_image, cv2.COLOR_RGB2GRAY)
        elif search_image.shape[2] == 4:
            search_image = cv2.cvtColor(search_image, cv2.COLOR_RGBA2GRAY)
    
    if reference_image.ndim == 3:
        if reference_image.shape[2] == 3:
            reference_image = cv2.cvtColor(reference_image, cv2.COLOR_RGB2GRAY)
        elif reference_image.shape[2] == 4:
            reference_image = cv2.cvtColor(reference_image, cv2.COLOR_RGBA2GRAY)
    
    # Extract local region around candidate
    h, w = search_image.shape
    ref_h, ref_w = reference_image.shape
    
    # Calculate region to extract (slightly larger than reference)
    margin = 5
    x0 = int(max(0, x - ref_w // 2 - margin))
    y0 = int(max(0, y - ref_h // 2 - margin))
    x1 = int(min(w, x + ref_w // 2 + margin))
    y1 = int(min(h, y + ref_h // 2 + margin))
    
    local_search = search_image[y0:y1, x0:x1]
    
    if local_search.size == 0 or local_search.shape[0] < ref_h or local_search.shape[1] < ref_w:
        return 0.0
    
    # Run template matching on local region
    try:
        result = match_template_local(local_search, reference_image, reference_scale=reference_scale)
        # The max_score from ZNCC is already in [0, 1] range
        return float(result.max_score)
    except Exception:
        return 0.0


def compute_global_appearance_score(
    search_image: np.ndarray,
    reference_image: np.ndarray,
    reference_scale: float = 1.0,
) -> Tuple[float, float, float]:
    """Compute global appearance score using full template matching.
    
    Args:
        search_image: Search image
        reference_image: Reference template
        reference_scale: Scale factor for reference
        
    Returns:
        Tuple of (max_score, x, y) for best match
    """
    try:
        result = match_template_local(search_image, reference_image, reference_scale=reference_scale)
        return float(result.max_score), float(result.center_x), float(result.center_y)
    except Exception:
        return 0.0, 0.0, 0.0
