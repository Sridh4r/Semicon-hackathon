"""Neighborhood scoring for DRAM structure verification."""

from __future__ import annotations

from typing import List

import numpy as np

from fingerprint.neighborhood_fingerprint import NeighborhoodFingerprint, compare_neighborhood_fingerprints


def compute_neighborhood_score(
    target_fingerprint: NeighborhoodFingerprint,
    candidate_fingerprints: List[NeighborhoodFingerprint],
) -> float:
    """Compute neighborhood similarity score for a candidate.
    
    Args:
        target_fingerprint: Target neighborhood fingerprint
        candidate_fingerprints: List of candidate fingerprints to compare against
        
    Returns:
        Neighborhood similarity score in [0, 1]
    """
    if not candidate_fingerprints:
        return 0.0
    
    # Compare with all candidates and take maximum similarity
    similarities = []
    for candidate_fp in candidate_fingerprints:
        similarity = compare_neighborhood_fingerprints(target_fingerprint, candidate_fp)
        similarities.append(similarity)
    
    return float(max(similarities))


def compute_batch_neighborhood_scores(
    target_fingerprint: NeighborhoodFingerprint,
    candidate_fingerprints: List[NeighborhoodFingerprint],
) -> List[float]:
    """Compute neighborhood scores for multiple candidates.
    
    Args:
        target_fingerprint: Target neighborhood fingerprint
        candidate_fingerprints: List of candidate fingerprints
        
    Returns:
        List of similarity scores
    """
    scores = []
    for candidate_fp in candidate_fingerprints:
        score = compute_neighborhood_score(target_fingerprint, [candidate_fp])
        scores.append(score)
    
    return scores
