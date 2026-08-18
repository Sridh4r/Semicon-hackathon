"""Row-Column Phase Fingerprint (RCPF) for DRAM lattice matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class RCPFResult:
    """Result of Row-Column Phase Fingerprint calculation."""
    
    phase_x: float
    phase_y: float
    row_phase: float
    column_phase: float
    row_pitch: float
    column_pitch: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "phase_x": self.phase_x,
            "phase_y": self.phase_y,
            "row_phase": self.row_phase,
            "column_phase": self.column_phase,
            "row_pitch": self.row_pitch,
            "column_pitch": self.column_pitch,
        }


def calculate_rcpf(
    x: float,
    y: float,
    row_pitch: float,
    column_pitch: float,
) -> RCPFResult:
    """Calculate Row-Column Phase Fingerprint for a position.
    
    The RCPF captures where a position falls within the DRAM lattice's
    periodic unit cell, which is crucial for distinguishing between
    visually similar but structurally different locations.
    
    Args:
        x: X coordinate in image space
        y: Y coordinate in image space
        row_pitch: Row pitch of DRAM lattice
        column_pitch: Column pitch of DRAM lattice
        
    Returns:
        RCPFResult containing phase information
    """
    # Calculate phases (position within periodic unit cell)
    phase_x = (x % column_pitch) / column_pitch
    phase_y = (y % row_pitch) / row_pitch
    
    # Normalize to [0, 1)
    phase_x = phase_x % 1.0
    phase_y = phase_y % 1.0
    
    # Calculate row and column phases (inverse of position phases)
    row_phase = phase_y
    column_phase = phase_x
    
    return RCPFResult(
        phase_x=phase_x,
        phase_y=phase_y,
        row_phase=row_phase,
        column_phase=column_phase,
        row_pitch=row_pitch,
        column_pitch=column_pitch,
    )


def compare_rcpf(
    rcpf1: RCPFResult,
    rcpf2: RCPFResult,
    phase_weight: float = 0.5,
    row_weight: float = 0.25,
    column_weight: float = 0.25,
) -> float:
    """Compare two RCPF results and return similarity score.
    
    Args:
        rcpf1: First RCPF result
        rcpf2: Second RCPF result
        phase_weight: Weight for overall phase similarity
        row_weight: Weight for row phase similarity
        column_weight: Weight for column phase similarity
        
    Returns:
        Similarity score in [0, 1] where 1 means identical
    """
    # Calculate phase differences (accounting for wraparound)
    def phase_distance(p1: float, p2: float) -> float:
        diff = abs(p1 - p2)
        return min(diff, 1.0 - diff)
    
    # Overall phase similarity
    phase_x_dist = phase_distance(rcpf1.phase_x, rcpf2.phase_x)
    phase_y_dist = phase_distance(rcpf1.phase_y, rcpf2.phase_y)
    phase_similarity = 1.0 - np.sqrt(phase_x_dist**2 + phase_y_dist**2) / np.sqrt(2)
    
    # Row phase similarity
    row_dist = phase_distance(rcpf1.row_phase, rcpf2.row_phase)
    row_similarity = 1.0 - row_dist
    
    # Column phase similarity
    col_dist = phase_distance(rcpf1.column_phase, rcpf2.column_phase)
    col_similarity = 1.0 - col_dist
    
    # Weighted combination
    total_weight = phase_weight + row_weight + column_weight
    if total_weight == 0:
        return 0.0
    
    similarity = (
        phase_weight * phase_similarity +
        row_weight * row_similarity +
        column_weight * col_similarity
    ) / total_weight
    
    return max(0.0, min(1.0, similarity))


def batch_calculate_rcpf(
    positions: list[tuple[float, float]],
    row_pitch: float,
    column_pitch: float,
) -> list[RCPFResult]:
    """Calculate RCPF for multiple positions efficiently.
    
    Args:
        positions: List of (x, y) tuples
        row_pitch: Row pitch of DRAM lattice
        column_pitch: Column pitch of DRAM lattice
        
    Returns:
        List of RCPFResult objects
    """
    results = []
    for x, y in positions:
        rcpf = calculate_rcpf(x, y, row_pitch, column_pitch)
        results.append(rcpf)
    return results


def find_best_phase_match(
    target_rcpf: RCPFResult,
    candidate_rcpfs: list[RCPFResult],
    min_similarity: float = 0.7,
) -> tuple[int, float]:
    """Find the candidate with best phase match to target.
    
    Args:
        target_rcpf: Target RCPF to match
        candidate_rcpfs: List of candidate RCPF results
        min_similarity: Minimum similarity threshold
        
    Returns:
        Tuple of (best_index, best_similarity) or (-1, 0.0) if no match
    """
    if not candidate_rcpfs:
        return -1, 0.0
    
    best_index = -1
    best_similarity = 0.0
    
    for i, candidate_rcpf in enumerate(candidate_rcpfs):
        similarity = compare_rcpf(target_rcpf, candidate_rcpf)
        if similarity > best_similarity:
            best_similarity = similarity
            best_index = i
    
    if best_similarity < min_similarity:
        return -1, 0.0
    
    return best_index, best_similarity


def rcpf_to_spatial_features(
    rcpf: RCPFResult,
    num_bins: int = 8,
) -> np.ndarray:
    """Convert RCPF to spatial feature vector for machine learning.
    
    Args:
        rcpf: RCPF result
        num_bins: Number of bins for phase discretization
        
    Returns:
        Feature vector of length 2 * num_bins
    """
    # Discretize phases into bins
    phase_x_bin = int(rcpf.phase_x * num_bins) % num_bins
    phase_y_bin = int(rcpf.phase_y * num_bins) % num_bins
    
    # Create one-hot encoding
    features = np.zeros(2 * num_bins, dtype=np.float32)
    features[phase_x_bin] = 1.0
    features[num_bins + phase_y_bin] = 1.0
    
    return features
