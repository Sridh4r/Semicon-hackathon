"""Top-K candidate selection for efficient processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Callable, Optional

import numpy as np

from .candidate_generator import MatchCandidate


@dataclass(frozen=True)
class TopKConfig:
    """Configuration for Top-K selection."""
    
    k: int = 20  # Number of top candidates to select
    scoring_function: Optional[str] = "combined"  # "edge_density", "phase", "combined"
    phase_weight: float = 0.3  # Weight for phase-based scoring
    edge_weight: float = 0.7  # Weight for edge density scoring


def select_top_k_candidates(
    candidates: List[MatchCandidate],
    config: Optional[TopKConfig] = None,
    target_phase_x: float = 0.5,
    target_phase_y: float = 0.5,
) -> List[MatchCandidate]:
    """Select top K candidates based on scoring function.
    
    Args:
        candidates: Input candidates
        config: Configuration for selection
        target_phase_x: Target phase for phase-based scoring
        target_phase_y: Target phase for phase-based scoring
        
    Returns:
        Top K candidates sorted by score
    """
    cfg = config or TopKConfig()
    
    if not candidates:
        return []
    
    # Score candidates based on configuration
    scored_candidates = _score_candidates(
        candidates, cfg, target_phase_x, target_phase_y
    )
    
    # Sort by score and select top K
    scored_candidates.sort(key=lambda c: c[1], reverse=True)
    top_k = [c[0] for c in scored_candidates[:cfg.k]]
    
    return top_k


def _score_candidates(
    candidates: List[MatchCandidate],
    config: TopKConfig,
    target_phase_x: float,
    target_phase_y: float,
) -> List[tuple[MatchCandidate, float]]:
    """Score candidates based on configuration.
    
    Args:
        candidates: Input candidates
        config: Configuration for scoring
        target_phase_x: Target phase for phase-based scoring
        target_phase_y: Target phase for phase-based scoring
        
    Returns:
        List of (candidate, score) tuples
    """
    scored: List[tuple[MatchCandidate, float]] = []
    
    for candidate in candidates:
        if config.scoring_function == "edge_density":
            score = candidate.edge_density
        elif config.scoring_function == "phase":
            score = _compute_phase_score(
                candidate, target_phase_x, target_phase_y
            )
        elif config.scoring_function == "combined":
            edge_score = candidate.edge_density
            phase_score = _compute_phase_score(
                candidate, target_phase_x, target_phase_y
            )
            score = (config.edge_weight * edge_score + 
                    config.phase_weight * phase_score)
        else:
            score = candidate.edge_density  # Default to edge density
        
        scored.append((candidate, score))
    
    return scored


def _compute_phase_score(
    candidate: MatchCandidate,
    target_phase_x: float,
    target_phase_y: float,
) -> float:
    """Compute phase similarity score.
    
    Args:
        candidate: Candidate to score
        target_phase_x: Target phase in x direction
        target_phase_y: Target phase in y direction
        
    Returns:
        Phase similarity score (higher = more similar)
    """
    dx = abs(candidate.phase_x - target_phase_x)
    dy = abs(candidate.phase_y - target_phase_y)
    
    # Handle wraparound for phases
    dx = min(dx, 1.0 - dx)
    dy = min(dy, 1.0 - dy)
    
    # Convert to similarity score (closer = higher score)
    phase_distance = np.sqrt(dx**2 + dy**2)
    max_distance = np.sqrt(2)  # Maximum possible phase distance
    
    similarity = 1.0 - (phase_distance / max_distance)
    
    return similarity


def adaptive_k_selection(
    candidates: List[MatchCandidate],
    base_k: int = 20,
    max_k: int = 50,
    score_variance_threshold: float = 0.1,
) -> int:
    """Adaptively select K based on score distribution.
    
    Args:
        candidates: Input candidates
        base_k: Base number of candidates
        max_k: Maximum number of candidates
        score_variance_threshold: Threshold for increasing K
        
    Returns:
        Adaptive K value
    """
    if len(candidates) <= base_k:
        return len(candidates)
    
    # Extract edge densities as simple scores
    scores = [c.edge_density for c in candidates]
    
    # Calculate variance
    if len(scores) > 1:
        variance = np.var(scores)
        if variance > score_variance_threshold:
            # High variance - use more candidates
            return min(max_k, int(base_k * 1.5))
    
    return base_k


def diverse_top_k_selection(
    candidates: List[MatchCandidate],
    k: int = 20,
    diversity_radius: float = 20.0,
) -> List[MatchCandidate]:
    """Select diverse top K candidates to avoid clustering.
    
    Args:
        candidates: Input candidates (should be sorted by priority)
        k: Number of candidates to select
        diversity_radius: Minimum distance between selected candidates
        
    Returns:
        Diverse top K candidates
    """
    if not candidates:
        return []
    
    diverse: List[MatchCandidate] = [candidates[0]]
    
    for candidate in candidates[1:]:
        if len(diverse) >= k:
            break
        
        # Check if candidate is far enough from selected candidates
        is_diverse = True
        for selected in diverse:
            dist = np.sqrt((candidate.x - selected.x)**2 + (candidate.y - selected.y)**2)
            if dist < diversity_radius:
                is_diverse = False
                break
        
        if is_diverse:
            diverse.append(candidate)
    
    return diverse
