"""Confidence estimation for DRAM-LOCX results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class ConfidenceResult:
    """Result of confidence estimation."""
    
    overall_confidence: float
    score_gap: float
    rcpf_consistency: float
    neighborhood_consistency: float
    topology_consistency: float
    num_candidates: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "overall_confidence": self.overall_confidence,
            "score_gap": self.score_gap,
            "rcpf_consistency": self.rcpf_consistency,
            "neighborhood_consistency": self.neighborhood_consistency,
            "topology_consistency": self.topology_consistency,
            "num_candidates": self.num_candidates,
        }


def compute_confidence(
    best_score: float,
    all_scores: List[float],
    rcpf_consistency: float = 0.8,
    neighborhood_consistency: float = 0.8,
    topology_consistency: float = 0.8,
) -> ConfidenceResult:
    """Compute confidence metric for localization result.
    
    Args:
        best_score: Score of best candidate
        all_scores: Scores of all candidates
        rcpf_consistency: RCPF consistency metric
        neighborhood_consistency: Neighborhood consistency metric
        topology_consistency: Topology consistency metric
        
    Returns:
        ConfidenceResult with confidence metrics
    """
    if not all_scores:
        return ConfidenceResult(
            overall_confidence=0.0,
            score_gap=0.0,
            rcpf_consistency=rcpf_consistency,
            neighborhood_consistency=neighborhood_consistency,
            topology_consistency=topology_consistency,
            num_candidates=0,
        )
    
    # Sort scores
    sorted_scores = sorted(all_scores, reverse=True)
    
    # Compute score gap between best and second best
    if len(sorted_scores) >= 2:
        score_gap = sorted_scores[0] - sorted_scores[1]
    else:
        score_gap = 1.0  # Only one candidate - high confidence
    
    # Score quality component
    score_quality = best_score
    
    # Score gap component (larger gap = higher confidence)
    gap_component = min(1.0, score_gap * 2.0)  # Amplify gap effect
    
    # Consistency components
    consistency_avg = (rcpf_consistency + neighborhood_consistency + topology_consistency) / 3.0
    
    # Number of candidates component (more candidates considered = more robust)
    num_candidates = len(all_scores)
    candidate_robustness = min(1.0, num_candidates / 20.0)  # Normalize to [0, 1]
    
    # Combine components
    overall_confidence = (
        0.4 * score_quality +
        0.3 * gap_component +
        0.2 * consistency_avg +
        0.1 * candidate_robustness
    )
    
    return ConfidenceResult(
        overall_confidence=overall_confidence,
        score_gap=score_gap,
        rcpf_consistency=rcpf_consistency,
        neighborhood_consistency=neighborhood_consistency,
        topology_consistency=topology_consistency,
        num_candidates=num_candidates,
    )


def is_high_confidence(confidence_result: ConfidenceResult, threshold: float = 0.8) -> bool:
    """Check if result has high confidence.
    
    Args:
        confidence_result: Confidence result to check
        threshold: Confidence threshold
        
    Returns:
        True if confidence is above threshold
    """
    return confidence_result.overall_confidence >= threshold


def is_ambiguous_confidence(confidence_result: ConfidenceResult, threshold: float = 0.6) -> bool:
    """Check if result has ambiguous confidence.
    
    Args:
        confidence_result: Confidence result to check
        threshold: Low confidence threshold
        
    Returns:
        True if confidence is below threshold
    """
    return confidence_result.overall_confidence < threshold
