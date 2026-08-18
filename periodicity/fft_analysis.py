"""FFT-based periodicity analysis for DRAM structure detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from .frequency_peaks import detect_frequency_peaks, find_symmetric_peak_pairs, FrequencyPeak


@dataclass(frozen=True)
class PeriodicityResult:
    """Result of periodicity analysis."""
    
    row_period: float
    column_period: float
    orientation: float
    confidence: float
    frequency_peaks: List[FrequencyPeak]
    symmetric_pairs: int
    method: str = "fft"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "row_period": self.row_period,
            "column_period": self.column_period,
            "orientation": self.orientation,
            "confidence": self.confidence,
            "frequency_peaks_count": len(self.frequency_peaks),
            "symmetric_pairs": self.symmetric_pairs,
            "method": self.method,
        }


def analyze_periodicity(
    image: np.ndarray,
    min_peak_height: float = 0.15,
    min_distance: int = 5,
    max_peaks: int = 20,
    apply_window: bool = True,
) -> PeriodicityResult:
    """Analyze image periodicity using 2D FFT.
    
    Args:
        image: Input image (grayscale or will be converted)
        min_peak_height: Minimum normalized peak height for detection
        min_distance: Minimum pixel distance between peaks
        max_peaks: Maximum number of peaks to detect
        apply_window: Whether to apply Hanning window to reduce edge effects
        
    Returns:
        PeriodicityResult containing estimated periods and confidence
    """
    # Convert to grayscale if needed
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    
    # Ensure float type for FFT
    image_float = image.astype(np.float32)
    
    # Remove mean
    image_float = image_float - image_float.mean()
    
    # Apply window function to reduce edge artifacts
    if apply_window:
        h, w = image.shape
        window_y = np.hanning(h)
        window_x = np.hanning(w)
        window = np.outer(window_y, window_x)
        image_float = image_float * window
    
    # Compute 2D FFT
    fft = np.fft.fft2(image_float)
    
    # Shift zero frequency to center
    fft_shifted = np.fft.fftshift(fft)
    
    # Compute magnitude spectrum
    magnitude_spectrum = np.abs(fft_shifted)
    
    # Log transform for better visualization/peak detection
    log_spectrum = np.log1p(magnitude_spectrum)
    
    # Detect frequency peaks
    peaks = detect_frequency_peaks(
        log_spectrum,
        min_peak_height=min_peak_height,
        min_distance=min_distance,
        max_peaks=max_peaks,
    )
    
    # Find symmetric pairs (indicates true periodicity)
    symmetric_pairs = find_symmetric_peak_pairs(peaks, tolerance=0.03)
    
    # Estimate periods from peaks
    if len(peaks) < 2:
        # Fallback to default values if insufficient peaks
        h, w = image.shape
        return PeriodicityResult(
            row_period=float(h / 16),
            column_period=float(w / 16),
            orientation=0.0,
            confidence=0.1,
            frequency_peaks=peaks,
            symmetric_pairs=len(symmetric_pairs),
        )
    
    # Use symmetric pairs for robust estimation
    row_period, column_period, orientation, confidence = _estimate_periods_from_peaks(
        peaks, symmetric_pairs, image.shape
    )
    
    return PeriodicityResult(
        row_period=row_period,
        column_period=column_period,
        orientation=orientation,
        confidence=confidence,
        frequency_peaks=peaks,
        symmetric_pairs=len(symmetric_pairs),
    )


def _estimate_periods_from_peaks(
    peaks: List[FrequencyPeak],
    symmetric_pairs: List[tuple],
    image_shape: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Estimate row/column periods and orientation from frequency peaks.
    
    Args:
        peaks: Detected frequency peaks
        symmetric_pairs: Symmetric peak pairs
        image_shape: Shape of input image (height, width)
        
    Returns:
        (row_period, column_period, orientation, confidence)
    """
    h, w = image_shape
    
    if len(symmetric_pairs) >= 2:
        # Use symmetric pairs for robust estimation
        periods_x = []
        periods_y = []
        
        for p1, p2 in symmetric_pairs:
            # Convert frequency to period
            if abs(p1.frequency_x) > 0.01:
                period_x = 1.0 / abs(p1.frequency_x)
                periods_x.append(period_x)
            if abs(p1.frequency_y) > 0.01:
                period_y = 1.0 / abs(p1.frequency_y)
                periods_y.append(period_y)
        
        if periods_x and periods_y:
            # Use median for robustness
            col_period = float(np.median(periods_x))
            row_period = float(np.median(periods_y))
            
            # Clamp to reasonable ranges
            col_period = max(5.0, min(col_period, w / 4))
            row_period = max(5.0, min(row_period, h / 4))
            
            # Estimate orientation from dominant peak
            dominant_pair = max(symmetric_pairs, key=lambda pair: pair[0].magnitude_normalized)
            dominant_peak = dominant_pair[0]
            orientation = float(np.degrees(np.arctan2(
                dominant_peak.frequency_y,
                dominant_peak.frequency_x
            )))
            
            confidence = min(0.95, 0.5 + 0.1 * len(symmetric_pairs))
            
            return row_period, col_period, orientation, confidence
    
    # Fallback: use all peaks
    periods_x = []
    periods_y = []
    
    for peak in peaks[:5]:  # Use top 5 peaks
        if abs(peak.frequency_x) > 0.01:
            period_x = 1.0 / abs(peak.frequency_x)
            periods_x.append(period_x)
        if abs(peak.frequency_y) > 0.01:
            period_y = 1.0 / abs(peak.frequency_y)
            periods_y.append(period_y)
    
    if periods_x and periods_y:
        col_period = float(np.median(periods_x))
        row_period = float(np.median(periods_y))
        
        col_period = max(5.0, min(col_period, w / 4))
        row_period = max(5.0, min(row_period, h / 4))
        
        orientation = 0.0
        confidence = 0.6
        
        return row_period, col_period, orientation, confidence
    
    # Ultimate fallback
    return float(h / 16), float(w / 16), 0.0, 0.3
