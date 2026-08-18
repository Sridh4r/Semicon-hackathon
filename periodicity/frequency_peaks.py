"""Frequency peak detection from FFT magnitude spectrum."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import cv2


@dataclass(frozen=True)
class FrequencyPeak:
    """Represents a detected frequency peak."""
    
    x: int
    y: int
    magnitude: float
    frequency_x: float
    frequency_y: float
    magnitude_normalized: float


def detect_frequency_peaks(
    magnitude_spectrum: np.ndarray,
    min_peak_height: float = 0.1,
    min_distance: int = 5,
    max_peaks: int = 20,
) -> List[FrequencyPeak]:
    """Detect dominant frequency peaks in the magnitude spectrum.
    
    Args:
        magnitude_spectrum: 2D FFT magnitude spectrum (zero-frequency centered)
        min_peak_height: Minimum normalized peak height (0-1)
        min_distance: Minimum pixel distance between peaks
        max_peaks: Maximum number of peaks to return
        
    Returns:
        List of FrequencyPeak objects sorted by magnitude (descending)
    """
    if magnitude_spectrum.ndim != 2:
        raise ValueError("magnitude_spectrum must be 2D")
    
    h, w = magnitude_spectrum.shape
    
    # Normalize spectrum to [0, 1]
    max_mag = magnitude_spectrum.max()
    if max_mag <= 0:
        return []
    
    normalized = magnitude_spectrum / max_mag
    
    # Suppress DC component (center of spectrum)
    center_y, center_x = h // 2, w // 2
    dc_radius = max(3, min(h, w) // 20)
    
    y_grid, x_grid = np.ogrid[:h, :w]
    dc_mask = (x_grid - center_x)**2 + (y_grid - center_y)**2 < dc_radius**2
    normalized[dc_mask] = 0
    
    # Find peaks using local maximum filter (using OpenCV dilate)
    kernel_size = min_distance * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    local_max = cv2.dilate(normalized, kernel)
    peak_mask = (normalized == local_max) & (normalized >= min_peak_height)
    
    # Get peak coordinates
    peak_coords = np.argwhere(peak_mask)
    
    if len(peak_coords) == 0:
        return []
    
    # Create peak objects
    peaks: List[FrequencyPeak] = []
    for y, x in peak_coords:
        mag = float(normalized[y, x])
        
        # Convert to frequency (accounting for FFT shift)
        freq_x = (x - center_x) / w if x != center_x else 0.0
        freq_y = (y - center_y) / h if y != center_y else 0.0
        
        peaks.append(FrequencyPeak(
            x=int(x),
            y=int(y),
            magnitude=float(magnitude_spectrum[y, x]),
            frequency_x=float(freq_x),
            frequency_y=float(freq_y),
            magnitude_normalized=mag,
        ))
    
    # Sort by magnitude and return top K
    peaks.sort(key=lambda p: p.magnitude_normalized, reverse=True)
    return peaks[:max_peaks]


def find_symmetric_peak_pairs(
    peaks: List[FrequencyPeak],
    tolerance: float = 0.05,
) -> List[Tuple[FrequencyPeak, FrequencyPeak]]:
    """Find symmetric peak pairs around the DC component.
    
    This helps identify true periodic structures which produce symmetric
    frequency patterns.
    
    Args:
        peaks: List of detected peaks
        tolerance: Frequency tolerance for symmetry matching
        
    Returns:
        List of symmetric peak pairs
    """
    pairs: List[Tuple[FrequencyPeak, FrequencyPeak]] = []
    
    for i, p1 in enumerate(peaks):
        for p2 in peaks[i+1:]:
            # Check if peaks are approximately symmetric around origin
            freq_x_sum = abs(p1.frequency_x + p2.frequency_x)
            freq_y_sum = abs(p1.frequency_y + p2.frequency_y)
            
            if freq_x_sum < tolerance and freq_y_sum < tolerance:
                pairs.append((p1, p2))
    
    return pairs
