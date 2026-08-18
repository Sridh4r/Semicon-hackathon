"""3×3 DRAM neighborhood fingerprint for distinguishing similar structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import cv2


@dataclass(frozen=True)
class NeighborhoodFingerprint:
    """Fingerprint of 3×3 DRAM neighborhood around a position."""
    
    center_features: dict
    neighbor_features: List[dict]
    relative_positions: List[Tuple[int, int]]
    neighborhood_consistency: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "center_features": self.center_features,
            "neighbor_features": self.neighbor_features,
            "relative_positions": self.relative_positions,
            "neighborhood_consistency": self.neighborhood_consistency,
        }


def compute_neighborhood_fingerprint(
    image: np.ndarray,
    x: float,
    y: float,
    row_pitch: float,
    column_pitch: float,
    neighborhood_size: int = 3,
    patch_size: int = 16,
) -> NeighborhoodFingerprint:
    """Compute 3×3 neighborhood fingerprint for a position.
    
    This captures the structural context around a candidate, which is
    crucial for distinguishing between visually similar but structurally
    different DRAM locations.
    
    Args:
        image: Input image (grayscale)
        x: X coordinate of center position
        y: Y coordinate of center position
        row_pitch: Row pitch of DRAM lattice
        column_pitch: Column pitch of DRAM lattice
        neighborhood_size: Size of neighborhood (default 3×3)
        patch_size: Size of patches to extract for each neighbor
        
    Returns:
        NeighborhoodFingerprint containing neighborhood structural information
    """
    # Convert to grayscale if needed
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    
    # Generate neighborhood positions
    neighbor_offsets = _generate_neighborhood_offsets(neighborhood_size)
    
    # Extract features for center and neighbors
    center_features = _extract_patch_features(image, x, y, patch_size)
    
    neighbor_features = []
    relative_positions = []
    
    for dx, dy in neighbor_offsets:
        nx = x + dx * column_pitch
        ny = y + dy * row_pitch
        
        # Check if neighbor is within image bounds
        h, w = image.shape
        if 0 <= nx < w and 0 <= ny < h:
            features = _extract_patch_features(image, nx, ny, patch_size)
            neighbor_features.append(features)
            relative_positions.append((dx, dy))
    
    # Calculate neighborhood consistency
    neighborhood_consistency = _calculate_neighborhood_consistency(
        center_features, neighbor_features
    )
    
    return NeighborhoodFingerprint(
        center_features=center_features,
        neighbor_features=neighbor_features,
        relative_positions=relative_positions,
        neighborhood_consistency=neighborhood_consistency,
    )


def _generate_neighborhood_offsets(size: int) -> List[Tuple[int, int]]:
    """Generate relative offsets for neighborhood positions.
    
    Args:
        size: Size of neighborhood (size × size)
        
    Returns:
        List of (dx, dy) offsets excluding center
    """
    offsets = []
    center = size // 2
    
    for dy in range(size):
        for dx in range(size):
            if dx == center and dy == center:
                continue  # Skip center
            offsets.append((dx - center, dy - center))
    
    return offsets


def _extract_patch_features(
    image: np.ndarray,
    x: float,
    y: float,
    patch_size: int,
) -> dict:
    """Extract features from a patch around a position.
    
    Args:
        image: Input image
        x: X coordinate of patch center
        y: Y coordinate of patch center
        patch_size: Size of patch to extract
        
    Returns:
        Dictionary of patch features
    """
    h, w = image.shape
    half_size = patch_size // 2
    
    # Extract patch with boundary handling
    x0 = int(max(0, x - half_size))
    y0 = int(max(0, y - half_size))
    x1 = int(min(w, x + half_size))
    y1 = int(min(h, y + half_size))
    
    patch = image[y0:y1, x0:x1]
    
    if patch.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "edge_density": 0.0,
            "gradient_magnitude": 0.0,
        }
    
    # Basic statistics
    mean_val = float(np.mean(patch))
    std_val = float(np.std(patch))
    
    # Edge density
    grad_x = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    edge_density = float(np.mean(gradient_magnitude))
    
    return {
        "mean": mean_val,
        "std": std_val,
        "edge_density": edge_density,
        "gradient_magnitude": float(np.mean(gradient_magnitude)),
    }


def _calculate_neighborhood_consistency(
    center_features: dict,
    neighbor_features: List[dict],
) -> float:
    """Calculate consistency between center and neighbors.
    
    Args:
        center_features: Features of center patch
        neighbor_features: Features of neighbor patches
        
    Returns:
        Consistency score in [0, 1]
    """
    if not neighbor_features:
        return 0.0
    
    # Calculate feature differences
    consistencies = []
    
    for neighbor in neighbor_features:
        # Mean intensity consistency
        mean_diff = abs(center_features["mean"] - neighbor["mean"]) / 255.0
        mean_consistency = 1.0 - mean_diff
        
        # Edge density consistency
        edge_diff = abs(center_features["edge_density"] - neighbor["edge_density"])
        edge_diff = min(1.0, edge_diff / 50.0)  # Normalize
        edge_consistency = 1.0 - edge_diff
        
        # Combined consistency
        combined = (mean_consistency + edge_consistency) / 2.0
        consistencies.append(combined)
    
    # Return average consistency
    return float(np.mean(consistencies))


def compare_neighborhood_fingerprints(
    fp1: NeighborhoodFingerprint,
    fp2: NeighborhoodFingerprint,
    center_weight: float = 0.4,
    neighbor_weight: float = 0.4,
    consistency_weight: float = 0.2,
) -> float:
    """Compare two neighborhood fingerprints.
    
    Args:
        fp1: First neighborhood fingerprint
        fp2: Second neighborhood fingerprint
        center_weight: Weight for center feature similarity
        neighbor_weight: Weight for neighbor feature similarity
        consistency_weight: Weight for consistency similarity
        
    Returns:
        Similarity score in [0, 1]
    """
    # Center feature similarity
    center_sim = _compare_patch_features(fp1.center_features, fp2.center_features)
    
    # Neighbor feature similarity
    if len(fp1.neighbor_features) == 0 or len(fp2.neighbor_features) == 0:
        neighbor_sim = 0.0
    else:
        # Compare corresponding neighbors by relative position
        neighbor_sims = []
        for i, (pos1, features1) in enumerate(zip(fp1.relative_positions, fp1.neighbor_features)):
            # Find matching neighbor in fp2
            for j, (pos2, features2) in enumerate(zip(fp2.relative_positions, fp2.neighbor_features)):
                if pos1 == pos2:
                    sim = _compare_patch_features(features1, features2)
                    neighbor_sims.append(sim)
                    break
        
        neighbor_sim = float(np.mean(neighbor_sims)) if neighbor_sims else 0.0
    
    # Consistency similarity
    consistency_sim = 1.0 - abs(fp1.neighborhood_consistency - fp2.neighborhood_consistency)
    
    # Weighted combination
    total_weight = center_weight + neighbor_weight + consistency_weight
    similarity = (
        center_weight * center_sim +
        neighbor_weight * neighbor_sim +
        consistency_weight * consistency_sim
    ) / total_weight
    
    return max(0.0, min(1.0, similarity))


def _compare_patch_features(f1: dict, f2: dict) -> float:
    """Compare two patch feature dictionaries.
    
    Args:
        f1: First patch features
        f2: Second patch features
        
    Returns:
        Similarity score in [0, 1]
    """
    # Mean similarity
    mean_diff = abs(f1["mean"] - f2["mean"]) / 255.0
    mean_sim = 1.0 - mean_diff
    
    # Edge density similarity
    edge_diff = abs(f1["edge_density"] - f2["edge_density"])
    edge_diff = min(1.0, edge_diff / 50.0)
    edge_sim = 1.0 - edge_diff
    
    # Combined similarity
    return (mean_sim + edge_sim) / 2.0
