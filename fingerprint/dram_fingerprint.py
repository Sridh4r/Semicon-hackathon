"""DRAM fingerprint combining multiple structural features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import cv2

from .row_column_phase import RCPFResult, calculate_rcpf


@dataclass(frozen=True)
class DRAMFingerprint:
    """Comprehensive fingerprint for DRAM structure matching."""
    
    rcpf: RCPFResult
    edge_histogram: np.ndarray
    intensity_stats: dict
    texture_features: dict
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "rcpf": self.rcpf.to_dict(),
            "edge_histogram": self.edge_histogram.tolist(),
            "intensity_stats": self.intensity_stats,
            "texture_features": self.texture_features,
        }


def compute_dram_fingerprint(
    image: np.ndarray,
    x: float,
    y: float,
    row_pitch: float,
    column_pitch: float,
    patch_size: int = 32,
) -> DRAMFingerprint:
    """Compute comprehensive DRAM fingerprint for a position.
    
    Args:
        image: Input image (grayscale)
        x: X coordinate of center position
        y: Y coordinate of center position
        row_pitch: Row pitch of DRAM lattice
        column_pitch: Column pitch of DRAM lattice
        patch_size: Size of patch to extract around position
        
    Returns:
        DRAMFingerprint containing multiple structural features
    """
    # Convert to grayscale if needed
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    
    # Calculate RCPF
    rcpf = calculate_rcpf(x, y, row_pitch, column_pitch)
    
    # Extract patch around position
    h, w = image.shape
    half_size = patch_size // 2
    x0 = int(max(0, x - half_size))
    y0 = int(max(0, y - half_size))
    x1 = int(min(w, x + half_size))
    y1 = int(min(h, y + half_size))
    
    patch = image[y0:y1, x0:x1]
    
    # Compute edge histogram
    edge_histogram = _compute_edge_histogram(patch)
    
    # Compute intensity statistics
    intensity_stats = _compute_intensity_stats(patch)
    
    # Compute texture features
    texture_features = _compute_texture_features(patch)
    
    return DRAMFingerprint(
        rcpf=rcpf,
        edge_histogram=edge_histogram,
        intensity_stats=intensity_stats,
        texture_features=texture_features,
    )


def _compute_edge_histogram(patch: np.ndarray, num_bins: int = 8) -> np.ndarray:
    """Compute edge orientation histogram."""
    if patch.size == 0:
        return np.zeros(num_bins, dtype=np.float32)
    
    # Compute gradients
    grad_x = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute orientation
    orientations = np.arctan2(grad_y, grad_x)
    magnitudes = np.sqrt(grad_x**2 + grad_y**2)
    
    # Create histogram
    histogram = np.zeros(num_bins, dtype=np.float32)
    
    # Bin orientations (-pi to pi)
    bin_width = 2 * np.pi / num_bins
    for i in range(num_bins):
        bin_start = -np.pi + i * bin_width
        bin_end = bin_start + bin_width
        
        mask = (orientations >= bin_start) & (orientations < bin_end)
        histogram[i] = np.sum(magnitudes[mask])
    
    # Normalize
    if histogram.sum() > 0:
        histogram = histogram / histogram.sum()
    
    return histogram


def _compute_intensity_stats(patch: np.ndarray) -> dict:
    """Compute intensity statistics."""
    if patch.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
        }
    
    return {
        "mean": float(np.mean(patch)),
        "std": float(np.std(patch)),
        "min": float(np.min(patch)),
        "max": float(np.max(patch)),
        "median": float(np.median(patch)),
    }


def _compute_texture_features(patch: np.ndarray) -> dict:
    """Compute texture features using Local Binary Pattern approximation."""
    if patch.size == 0:
        return {
            "contrast": 0.0,
            "homogeneity": 0.0,
            "energy": 0.0,
        }
    
    # Simple texture features based on local variance using OpenCV
    # Local mean
    local_mean = cv2.blur(patch.astype(np.float32), (3, 3))
    
    # Local variance
    local_variance = cv2.blur((patch.astype(np.float32) - local_mean)**2, (3, 3))
    
    # Global texture statistics
    contrast = float(np.std(local_variance))
    homogeneity = float(1.0 / (1.0 + np.mean(local_variance)))
    energy = float(np.mean(local_variance**2))
    
    return {
        "contrast": contrast,
        "homogeneity": homogeneity,
        "energy": energy,
    }


def compare_dram_fingerprints(
    fp1: DRAMFingerprint,
    fp2: DRAMFingerprint,
    rcpf_weight: float = 0.4,
    edge_weight: float = 0.3,
    intensity_weight: float = 0.2,
    texture_weight: float = 0.1,
) -> float:
    """Compare two DRAM fingerprints.
    
    Args:
        fp1: First fingerprint
        fp2: Second fingerprint
        rcpf_weight: Weight for RCPF similarity
        edge_weight: Weight for edge histogram similarity
        intensity_weight: Weight for intensity similarity
        texture_weight: Weight for texture similarity
        
    Returns:
        Similarity score in [0, 1]
    """
    from .row_column_phase import compare_rcpf
    
    # RCPF similarity
    rcpf_sim = compare_rcpf(fp1.rcpf, fp2.rcpf)
    
    # Edge histogram similarity (chi-square distance)
    edge_sim = 1.0 - cv2.compareHist(
        fp1.edge_histogram.astype(np.float32),
        fp2.edge_histogram.astype(np.float32),
        cv2.HISTCMP_CHISQR
    ) / 2.0  # Normalize to [0, 1]
    
    # Intensity similarity
    mean_diff = abs(fp1.intensity_stats["mean"] - fp2.intensity_stats["mean"]) / 255.0
    intensity_sim = 1.0 - mean_diff
    
    # Texture similarity
    texture_diff = (
        abs(fp1.texture_features["contrast"] - fp2.texture_features["contrast"]) +
        abs(fp1.texture_features["homogeneity"] - fp2.texture_features["homogeneity"])
    ) / 2.0
    texture_sim = 1.0 - min(1.0, texture_diff)
    
    # Weighted combination
    total_weight = rcpf_weight + edge_weight + intensity_weight + texture_weight
    similarity = (
        rcpf_weight * rcpf_sim +
        edge_weight * edge_sim +
        intensity_weight * intensity_sim +
        texture_weight * texture_sim
    ) / total_weight
    
    return max(0.0, min(1.0, similarity))
