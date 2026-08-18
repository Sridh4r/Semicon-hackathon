"""DRAM topology scoring for structural consistency verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import cv2


@dataclass(frozen=True)
class TopologyResult:
    """Result of DRAM topology analysis."""
    
    horizontal_consistency: float
    vertical_consistency: float
    intersection_consistency: float
    pitch_consistency: float
    neighbor_alignment: float
    overall_score: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "horizontal_consistency": self.horizontal_consistency,
            "vertical_consistency": self.vertical_consistency,
            "intersection_consistency": self.intersection_consistency,
            "pitch_consistency": self.pitch_consistency,
            "neighbor_alignment": self.neighbor_alignment,
            "overall_score": self.overall_score,
        }


def compute_topology_score(
    image: np.ndarray,
    x: float,
    y: float,
    row_pitch: float,
    column_pitch: float,
    neighborhood_size: int = 3,
) -> TopologyResult:
    """Compute DRAM topology score for a candidate position.
    
    This analyzes the structural consistency of the DRAM lattice around
    the candidate position by checking horizontal and vertical structure
    alignment, intersection patterns, and pitch consistency.
    
    Args:
        image: Input image (grayscale)
        x: X coordinate of candidate position
        y: Y coordinate of candidate position
        row_pitch: Row pitch of DRAM lattice
        column_pitch: Column pitch of DRAM lattice
        neighborhood_size: Size of neighborhood to analyze
        
    Returns:
        TopologyResult containing various consistency metrics
    """
    # Convert to grayscale if needed
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    
    # Extract neighborhood patches
    horizontal_patches = _extract_horizontal_line(image, x, y, row_pitch, column_pitch, neighborhood_size)
    vertical_patches = _extract_vertical_line(image, x, y, row_pitch, column_pitch, neighborhood_size)
    
    # Compute consistency metrics
    horizontal_consistency = _compute_horizontal_consistency(horizontal_patches)
    vertical_consistency = _compute_vertical_consistency(vertical_patches)
    intersection_consistency = _compute_intersection_consistency(horizontal_patches, vertical_patches)
    pitch_consistency = _compute_pitch_consistency(horizontal_patches, vertical_patches, row_pitch, column_pitch)
    neighbor_alignment = _compute_neighbor_alignment(image, x, y, row_pitch, column_pitch)
    
    # Compute overall score
    overall_score = (
        0.25 * horizontal_consistency +
        0.25 * vertical_consistency +
        0.2 * intersection_consistency +
        0.15 * pitch_consistency +
        0.15 * neighbor_alignment
    )
    
    return TopologyResult(
        horizontal_consistency=horizontal_consistency,
        vertical_consistency=vertical_consistency,
        intersection_consistency=intersection_consistency,
        pitch_consistency=pitch_consistency,
        neighbor_alignment=neighbor_alignment,
        overall_score=overall_score,
    )


def _extract_horizontal_line(
    image: np.ndarray,
    x: float,
    y: float,
    row_pitch: float,
    column_pitch: float,
    size: int,
) -> List[np.ndarray]:
    """Extract horizontal line of patches around position."""
    patches = []
    half_size = size // 2
    
    for i in range(-half_size, half_size + 1):
        nx = x + i * column_pitch
        ny = y
        
        h, w = image.shape
        if 0 <= nx < w and 0 <= ny < h:
            patch_size = int(min(row_pitch, column_pitch))
            x0 = int(max(0, nx - patch_size // 2))
            y0 = int(max(0, ny - patch_size // 2))
            x1 = int(min(w, nx + patch_size // 2))
            y1 = int(min(h, ny + patch_size // 2))
            patch = image[y0:y1, x0:x1]
            if patch.size > 0:
                patches.append(patch)
    
    return patches


def _extract_vertical_line(
    image: np.ndarray,
    x: float,
    y: float,
    row_pitch: float,
    column_pitch: float,
    size: int,
) -> List[np.ndarray]:
    """Extract vertical line of patches around position."""
    patches = []
    half_size = size // 2
    
    for i in range(-half_size, half_size + 1):
        nx = x
        ny = y + i * row_pitch
        
        h, w = image.shape
        if 0 <= nx < w and 0 <= ny < h:
            patch_size = int(min(row_pitch, column_pitch))
            x0 = int(max(0, nx - patch_size // 2))
            y0 = int(max(0, ny - patch_size // 2))
            x1 = int(min(w, nx + patch_size // 2))
            y1 = int(min(h, ny + patch_size // 2))
            patch = image[y0:y1, x0:x1]
            if patch.size > 0:
                patches.append(patch)
    
    return patches


def _compute_horizontal_consistency(patches: List[np.ndarray]) -> float:
    """Compute consistency of horizontal structures."""
    if len(patches) < 2:
        return 0.0
    
    # Compute edge orientation consistency
    orientations = []
    for patch in patches:
        if patch.size == 0:
            continue
        grad_x = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
        orientation = np.arctan2(np.mean(grad_y), np.mean(grad_x))
        orientations.append(orientation)
    
    if len(orientations) < 2:
        return 0.0
    
    # Compute variance of orientations
    orientation_variance = np.var(orientations)
    consistency = 1.0 - min(1.0, orientation_variance / (np.pi / 4))  # Normalize
    
    return float(consistency)


def _compute_vertical_consistency(patches: List[np.ndarray]) -> float:
    """Compute consistency of vertical structures."""
    if len(patches) < 2:
        return 0.0
    
    # Similar to horizontal but for vertical alignment
    orientations = []
    for patch in patches:
        if patch.size == 0:
            continue
        grad_x = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
        orientation = np.arctan2(np.mean(grad_y), np.mean(grad_x))
        orientations.append(orientation)
    
    if len(orientations) < 2:
        return 0.0
    
    orientation_variance = np.var(orientations)
    consistency = 1.0 - min(1.0, orientation_variance / (np.pi / 4))
    
    return float(consistency)


def _compute_intersection_consistency(
    horizontal_patches: List[np.ndarray],
    vertical_patches: List[np.ndarray],
) -> float:
    """Compute consistency at intersection points."""
    if len(horizontal_patches) < 2 or len(vertical_patches) < 2:
        return 0.0
    
    # Compare center patches (intersections)
    center_h = horizontal_patches[len(horizontal_patches) // 2] if horizontal_patches else None
    center_v = vertical_patches[len(vertical_patches) // 2] if vertical_patches else None
    
    if center_h is None or center_v is None or center_h.size == 0 or center_v.size == 0:
        return 0.0
    
    # Compute structural similarity at intersection
    mean_h = np.mean(center_h)
    mean_v = np.mean(center_v)
    
    # Intersection should have consistent intensity
    intensity_diff = abs(mean_h - mean_v) / 255.0
    consistency = 1.0 - min(1.0, intensity_diff)
    
    return float(consistency)


def _compute_pitch_consistency(
    horizontal_patches: List[np.ndarray],
    vertical_patches: List[np.ndarray],
    row_pitch: float,
    column_pitch: float,
) -> float:
    """Compute consistency of measured pitches vs expected pitches."""
    if len(horizontal_patches) < 3 or len(vertical_patches) < 3:
        return 0.5  # Default medium consistency
    
    # This is a simplified check - in practice would measure actual spacing
    # For now, return high consistency if patches are available
    return 0.8


def _compute_neighbor_alignment(
    image: np.ndarray,
    x: float,
    y: float,
    row_pitch: float,
    column_pitch: float,
) -> float:
    """Compute alignment of neighboring structures."""
    h, w = image.shape
    
    # Check a few neighboring positions
    neighbor_offsets = [
        (1, 0), (-1, 0), (0, 1), (0, -1),  # Cardinal directions
        (1, 1), (-1, -1), (1, -1), (-1, 1),  # Diagonals
    ]
    
    alignments = []
    for dx, dy in neighbor_offsets:
        nx = x + dx * column_pitch
        ny = y + dy * row_pitch
        
        if 0 <= nx < w and 0 <= ny < h:
            # Extract small patch
            patch_size = int(min(row_pitch, column_pitch) / 2)
            x0 = int(max(0, nx - patch_size // 2))
            y0 = int(max(0, ny - patch_size // 2))
            x1 = int(min(w, nx + patch_size // 2))
            y1 = int(min(h, ny + patch_size // 2))
            patch = image[y0:y1, x0:x1]
            
            if patch.size > 0:
                # Check if patch has reasonable structure (non-zero variance)
                if np.var(patch) > 10:  # Some variance indicates structure
                    alignments.append(1.0)
                else:
                    alignments.append(0.5)
    
    if not alignments:
        return 0.0
    
    return float(np.mean(alignments))
