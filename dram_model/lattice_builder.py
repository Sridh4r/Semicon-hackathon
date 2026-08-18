"""DRAM lattice construction for candidate generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import math


@dataclass(frozen=True)
class LatticeCandidate:
    """Represents a candidate position in the DRAM lattice."""
    
    x: float
    y: float
    row_index: int
    column_index: int
    phase_x: float
    phase_y: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "x": self.x,
            "y": self.y,
            "row": self.row_index,
            "column": self.column_index,
            "phase_x": self.phase_x,
            "phase_y": self.phase_y,
        }


@dataclass
class DRAMLattice:
    """Represents the DRAM row-column lattice structure."""
    
    row_pitch: float
    column_pitch: float
    orientation: float
    origin_x: float
    origin_y: float
    num_rows: int
    num_columns: int
    candidates: List[LatticeCandidate]
    
    def get_candidate_at_position(self, x: float, y: float, tolerance: float = 2.0) -> Optional[LatticeCandidate]:
        """Find the lattice candidate closest to given position."""
        min_dist = float('inf')
        closest = None
        
        for candidate in self.candidates:
            dist = math.sqrt((candidate.x - x)**2 + (candidate.y - y)**2)
            if dist < min_dist and dist <= tolerance:
                min_dist = dist
                closest = candidate
        
        return closest
    
    def get_candidates_in_region(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> List[LatticeCandidate]:
        """Get all candidates within a rectangular region."""
        return [
            c for c in self.candidates
            if x_min <= c.x <= x_max and y_min <= c.y <= y_max
        ]


def build_dram_lattice(
    image_shape: Tuple[int, int],
    row_pitch: float,
    column_pitch: float,
    orientation: float = 0.0,
    margin: int = 10,
) -> DRAMLattice:
    """Build a DRAM lattice structure for candidate generation.
    
    Args:
        image_shape: Shape of the search image (height, width)
        row_pitch: Estimated row pitch in pixels
        column_pitch: Estimated column pitch in pixels
        orientation: Orientation angle in degrees (default 0)
        margin: Margin from image edges in pixels
        
    Returns:
        DRAMLattice containing all valid candidate positions
    """
    h, w = image_shape
    
    # Validate inputs
    if row_pitch <= 0 or column_pitch <= 0:
        raise ValueError("Row and column pitches must be positive")
    
    if row_pitch > h / 2 or column_pitch > w / 2:
        raise ValueError("Pitches are too large for image dimensions")
    
    # Convert orientation to radians
    orientation_rad = math.radians(orientation)
    
    # Calculate lattice origin (center of image)
    origin_x = w / 2.0
    origin_y = h / 2.0
    
    # Calculate rotation matrix
    cos_theta = math.cos(orientation_rad)
    sin_theta = math.sin(orientation_rad)
    
    # Generate lattice points
    candidates: List[LatticeCandidate] = []
    
    # Estimate number of rows and columns needed
    num_rows = int(h / row_pitch) + 2
    num_columns = int(w / column_pitch) + 2
    
    # Center the lattice around the origin
    row_offset = num_rows // 2
    col_offset = num_columns // 2
    
    for row_idx in range(-row_offset, row_offset + 1):
        for col_idx in range(-col_offset, col_offset + 1):
            # Calculate position in lattice coordinates
            lattice_x = col_idx * column_pitch
            lattice_y = row_idx * row_pitch
            
            # Apply rotation
            rotated_x = lattice_x * cos_theta - lattice_y * sin_theta
            rotated_y = lattice_x * sin_theta + lattice_y * cos_theta
            
            # Translate to image coordinates
            x = origin_x + rotated_x
            y = origin_y + rotated_y
            
            # Check if within image bounds with margin
            if (margin <= x < w - margin and margin <= y < h - margin):
                # Calculate phase (position within the periodic unit cell)
                phase_x = (x % column_pitch) / column_pitch
                phase_y = (y % row_pitch) / row_pitch
                
                candidates.append(LatticeCandidate(
                    x=float(x),
                    y=float(y),
                    row_index=row_idx + row_offset,
                    column_index=col_idx + col_offset,
                    phase_x=phase_x,
                    phase_y=phase_y,
                ))
    
    return DRAMLattice(
        row_pitch=row_pitch,
        column_pitch=column_pitch,
        orientation=orientation,
        origin_x=origin_x,
        origin_y=origin_y,
        num_rows=num_rows,
        num_columns=num_columns,
        candidates=candidates,
    )


def refine_lattice_with_reference(
    lattice: DRAMLattice,
    reference_size: Tuple[int, int],
    search_size: Tuple[int, int],
) -> DRAMLattice:
    """Refine lattice by removing candidates that can't fit the reference.
    
    Args:
        lattice: Initial DRAM lattice
        reference_size: Size of reference template (height, width)
        search_size: Size of search image (height, width)
        
    Returns:
        Refined lattice with only valid candidates
    """
    ref_h, ref_w = reference_size
    search_h, search_w = search_size
    
    # Filter candidates that can accommodate the reference
    valid_candidates = [
        c for c in lattice.candidates
        if (ref_w / 2 <= c.x <= search_w - ref_w / 2 and
            ref_h / 2 <= c.y <= search_h - ref_h / 2)
    ]
    
    return DRAMLattice(
        row_pitch=lattice.row_pitch,
        column_pitch=lattice.column_pitch,
        orientation=lattice.orientation,
        origin_x=lattice.origin_x,
        origin_y=lattice.origin_y,
        num_rows=lattice.num_rows,
        num_columns=lattice.num_columns,
        candidates=valid_candidates,
    )
