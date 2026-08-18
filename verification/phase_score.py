"""Phase scoring for RCPF-based verification."""

from __future__ import annotations

from typing import List

import numpy as np

from fingerprint.row_column_phase import RCPFResult, compare_rcpf


def compute_phase_score(
    target_rcpf: RCPFResult,
    candidate_rcpf: RCPFResult,
) -> float:
    """Compute phase similarity score using RCPF comparison.
    
    Args:
        target_rcpf: Target RCPF
        candidate_rcpf: Candidate RCPF
        
    Returns:
        Phase similarity score in [0, 1]
    """
    return compare_rcpf(target_rcpf, candidate_rcpf)


def compute_batch_phase_scores(
    target_rcpf: RCPFResult,
    candidate_rcpfs: List[RCPFResult],
) -> List[float]:
    """Compute phase scores for multiple candidates.
    
    Args:
        target_rcpf: Target RCPF
        candidate_rcpfs: List of candidate RCPFs
        
    Returns:
        List of phase similarity scores
    """
    scores = []
    for candidate_rcpf in candidate_rcpfs:
        score = compute_phase_score(target_rcpf, candidate_rcpf)
        scores.append(score)
    
    return scores
