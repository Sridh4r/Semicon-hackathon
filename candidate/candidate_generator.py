"""Candidate generation from DRAM lattice for efficient search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import cv2

from dram_model.lattice_builder import DRAMLattice, LatticeCandidate


@dataclass(frozen=True)
class CandidateGeneratorConfig:
    """Configuration for candidate generation."""
    
    max_candidates: int = 500  # Maximum number of candidates to generate
    min_spacing: float = 5.0  # Minimum spacing between candidates
    use_edge_density: bool = True  # Use edge density to prioritize candidates
    edge_density_window: int = 15  # Window size for edge density calculation


@dataclass
class MatchCandidate:
    """Extended candidate with matching information."""
    
    x: float
    y: float
    row_index: int
    column_index: int
    phase_x: float
    phase_y: float
    edge_density: float = 0.0
    zncc_score: float = 0.0
    rcpf_score: float = 0.0
    neighborhood_score: float = 0.0
    topology_score: float = 0.0
    final_score: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "x": self.x,
            "y": self.y,
            "row": self.row_index,
            "column": self.column_index,
            "phase_x": self.phase_x,
            "phase_y": self.phase_y,
            "edge_density": self.edge_density,
            "zncc_score": self.zncc_score,
            "rcpf_score": self.rcpf_score,
            "neighborhood_score": self.neighborhood_score,
            "topology_score": self.topology_score,
            "final_score": self.final_score,
        }


def generate_candidates(
    search_image: np.ndarray,
    lattice: DRAMLattice,
    reference_size: Tuple[int, int],
    config: Optional[CandidateGeneratorConfig] = None,
) -> List[MatchCandidate]:
    """Generate match candidates from DRAM lattice.
    
    Args:
        search_image: Search image (grayscale)
        lattice: DRAM lattice structure
        reference_size: Size of reference template (height, width)
        config: Optional configuration for candidate generation
        
    Returns:
        List of MatchCandidate objects sorted by priority
    """
    cfg = config or CandidateGeneratorConfig()
    
    # Convert to grayscale if needed
    if search_image.ndim == 3:
        if search_image.shape[2] == 3:
            search_image = cv2.cvtColor(search_image, cv2.COLOR_RGB2GRAY)
        elif search_image.shape[2] == 4:
            search_image = cv2.cvtColor(search_image, cv2.COLOR_RGBA2GRAY)
    
    # Calculate edge density for candidate prioritization
    edge_density_map = None
    if cfg.use_edge_density:
        edge_density_map = _compute_edge_density(search_image, cfg.edge_density_window)
    
    # Convert lattice candidates to match candidates
    match_candidates: List[MatchCandidate] = []
    
    for lattice_candidate in lattice.candidates:
        # Calculate edge density at candidate position
        edge_density = 0.0
        if edge_density_map is not None:
            x, y = int(lattice_candidate.x), int(lattice_candidate.y)
            h, w = edge_density_map.shape
            if 0 <= y < h and 0 <= x < w:
                edge_density = float(edge_density_map[y, x])
        
        match_candidate = MatchCandidate(
            x=lattice_candidate.x,
            y=lattice_candidate.y,
            row_index=lattice_candidate.row_index,
            column_index=lattice_candidate.column_index,
            phase_x=lattice_candidate.phase_x,
            phase_y=lattice_candidate.phase_y,
            edge_density=edge_density,
        )
        
        match_candidates.append(match_candidate)
    
    # Sort by edge density (higher density = more likely to contain structure)
    if cfg.use_edge_density:
        match_candidates.sort(key=lambda c: c.edge_density, reverse=True)
    
    # Apply spacing constraint to avoid too many similar candidates
    if cfg.min_spacing > 0:
        match_candidates = _apply_spacing_constraint(
            match_candidates, cfg.min_spacing, cfg.max_candidates
        )
    else:
        # Just limit to max candidates
        match_candidates = match_candidates[:cfg.max_candidates]
    
    return match_candidates


def _compute_edge_density(image: np.ndarray, window_size: int) -> np.ndarray:
    """Compute edge density map for candidate prioritization.
    
    Args:
        image: Input grayscale image
        window_size: Size of sliding window for density calculation
        
    Returns:
        Edge density map
    """
    # Compute gradients
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute gradient magnitude
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize
    if grad_mag.max() > 0:
        grad_mag = grad_mag / grad_mag.max()
    
    # Apply local sum filter for density
    kernel = np.ones((window_size, window_size), dtype=np.float32)
    kernel = kernel / np.sum(kernel)
    
    edge_density = cv2.filter2D(grad_mag, -1, kernel)
    
    return edge_density


def _apply_spacing_constraint(
    candidates: List[MatchCandidate],
    min_spacing: float,
    max_candidates: int,
) -> List[MatchCandidate]:
    """Apply minimum spacing constraint to reduce redundant candidates.
    
    Args:
        candidates: Input candidates (should be sorted by priority)
        min_spacing: Minimum spacing between candidates
        max_candidates: Maximum number of candidates to return
        
    Returns:
        Filtered candidates with spacing constraint applied
    """
    if not candidates:
        return []
    
    filtered: List[MatchCandidate] = [candidates[0]]
    
    for candidate in candidates[1:]:
        # Check if candidate is far enough from all selected candidates
        is_valid = True
        for selected in filtered:
            dist = np.sqrt((candidate.x - selected.x)**2 + (candidate.y - selected.y)**2)
            if dist < min_spacing:
                is_valid = False
                break
        
        if is_valid:
            filtered.append(candidate)
            if len(filtered) >= max_candidates:
                break
    
    return filtered


def prioritize_candidates_by_phase(
    candidates: List[MatchCandidate],
    target_phase_x: float,
    target_phase_y: float,
    phase_tolerance: float = 0.1,
) -> List[MatchCandidate]:
    """Prioritize candidates based on phase similarity to target.
    
    Args:
        candidates: Input candidates
        target_phase_x: Target phase in x direction
        target_phase_y: Target phase in y direction
        phase_tolerance: Tolerance for phase matching
        
    Returns:
        Candidates sorted by phase similarity
    """
    def phase_distance(candidate: MatchCandidate) -> float:
        dx = abs(candidate.phase_x - target_phase_x)
        dy = abs(candidate.phase_y - target_phase_y)
        # Handle wraparound for phases
        dx = min(dx, 1.0 - dx)
        dy = min(dy, 1.0 - dy)
        return np.sqrt(dx**2 + dy**2)
    
    # Filter candidates within phase tolerance
    filtered = [
        c for c in candidates
        if phase_distance(c) <= phase_tolerance
    ]
    
    # Sort by phase distance
    filtered.sort(key=phase_distance)
    
    return filtered
