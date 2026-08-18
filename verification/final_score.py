"""Final scoring system combining all verification components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ScoreWeights:
    """Weights for different scoring components."""
    
    appearance: float = 0.35
    rcpf: float = 0.25
    neighborhood: float = 0.20
    topology: float = 0.15
    geometry: float = 0.05
    
    def __post_init__(self):
        """Validate that weights sum to 1.0."""
        total = self.appearance + self.rcpf + self.neighborhood + self.topology + self.geometry
        if not (0.99 <= total <= 1.01):  # Allow small floating point errors
            raise ValueError(f"Score weights must sum to 1.0, got {total}")


@dataclass(frozen=True)
class FinalScoreResult:
    """Result of final scoring computation."""
    
    final_score: float
    appearance_score: float
    rcpf_score: float
    neighborhood_score: float
    topology_score: float
    geometry_score: float
    weights: ScoreWeights
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "final_score": self.final_score,
            "appearance_score": self.appearance_score,
            "rcpf_score": self.rcpf_score,
            "neighborhood_score": self.neighborhood_score,
            "topology_score": self.topology_score,
            "geometry_score": self.geometry_score,
            "weights": {
                "appearance": self.weights.appearance,
                "rcpf": self.weights.rcpf,
                "neighborhood": self.weights.neighborhood,
                "topology": self.weights.topology,
                "geometry": self.weights.geometry,
            },
        }


def compute_final_score(
    appearance_score: float,
    rcpf_score: float,
    neighborhood_score: float,
    topology_score: float,
    geometry_score: float = 0.5,
    weights: Optional[ScoreWeights] = None,
) -> FinalScoreResult:
    """Compute final weighted score from individual component scores.
    
    Args:
        appearance_score: Appearance similarity score [0, 1]
        rcpf_score: RCPF similarity score [0, 1]
        neighborhood_score: Neighborhood similarity score [0, 1]
        topology_score: Topology consistency score [0, 1]
        geometry_score: Geometric prior score [0, 1]
        weights: Optional custom weights (uses default if None)
        
    Returns:
        FinalScoreResult with weighted combination
    """
    cfg = weights or ScoreWeights()
    
    # Ensure all scores are in [0, 1]
    appearance_score = max(0.0, min(1.0, appearance_score))
    rcpf_score = max(0.0, min(1.0, rcpf_score))
    neighborhood_score = max(0.0, min(1.0, neighborhood_score))
    topology_score = max(0.0, min(1.0, topology_score))
    geometry_score = max(0.0, min(1.0, geometry_score))
    
    # Compute weighted sum
    final_score = (
        cfg.appearance * appearance_score +
        cfg.rcpf * rcpf_score +
        cfg.neighborhood * neighborhood_score +
        cfg.topology * topology_score +
        cfg.geometry * geometry_score
    )
    
    return FinalScoreResult(
        final_score=final_score,
        appearance_score=appearance_score,
        rcpf_score=rcpf_score,
        neighborhood_score=neighborhood_score,
        topology_score=topology_score,
        geometry_score=geometry_score,
        weights=cfg,
    )


def compute_geometry_score(
    x: float,
    y: float,
    image_width: int,
    image_height: int,
    preferred_center: bool = False,
) -> float:
    """Compute geometric prior score based on position.
    
    Args:
        x: X coordinate
        y: Y coordinate
        image_width: Image width
        image_height: Image height
        preferred_center: Whether to prefer center positions
        
    Returns:
        Geometry score in [0, 1]
    """
    if not preferred_center:
        return 0.5  # Neutral if no preference
    
    # Compute distance from center
    center_x = image_width / 2.0
    center_y = image_height / 2.0
    max_distance = np.sqrt(center_x**2 + center_y**2)
    
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Convert to score (closer to center = higher score)
    geometry_score = 1.0 - (distance / max_distance)
    
    return float(max(0.0, min(1.0, geometry_score)))


def create_custom_weights(
    appearance: float = 0.35,
    rcpf: float = 0.25,
    neighborhood: float = 0.20,
    topology: float = 0.15,
    geometry: float = 0.05,
) -> ScoreWeights:
    """Create custom score weights.
    
    Args:
        appearance: Weight for appearance score
        rcpf: Weight for RCPF score
        neighborhood: Weight for neighborhood score
        topology: Weight for topology score
        geometry: Weight for geometry score
        
    Returns:
        ScoreWeights object
    """
    return ScoreWeights(
        appearance=appearance,
        rcpf=rcpf,
        neighborhood=neighborhood,
        topology=topology,
        geometry=geometry,
    )
