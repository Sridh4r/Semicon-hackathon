"""Autocorrelation-based periodicity analysis (fallback method)."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def compute_autocorrelation(image: np.ndarray) -> np.ndarray:
    """Compute 2D autocorrelation of an image using FFT.
    
    Args:
        image: Input image (grayscale or will be converted)
        
    Returns:
        2D autocorrelation map, zero-centered
    """
    # Convert to grayscale if needed
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    
    # Normalize to [0, 1]
    if image.max() > 0:
        image = image.astype(np.float32) / image.max()
    
    # Compute 2D autocorrelation using FFT
    fft = np.fft.fft2(image)
    autocorr = np.real(np.fft.ifft2(fft * np.conj(fft)))
    autocorr = np.fft.fftshift(autocorr)
    
    # Normalize
    if autocorr.max() > 0:
        autocorr = autocorr / autocorr.max()
    
    return autocorr


def estimate_period_from_autocorr_1d(
    signal: np.ndarray,
    center_idx: int,
    min_period: int = 3,
    max_period: Optional[int] = None,
) -> Optional[float]:
    """Estimate period from 1D autocorrelation signal.
    
    Args:
        signal: 1D autocorrelation signal
        center_idx: Index of DC component (center)
        min_period: Minimum period to consider
        max_period: Maximum period to consider (None = signal length / 3)
        
    Returns:
        Estimated period or None if estimation fails
    """
    if max_period is None:
        max_period = len(signal) // 3
    
    # Look at right side only (autocorrelation is symmetric)
    right_side = signal[center_idx + min_period:center_idx + max_period]
    
    if len(right_side) < min_period * 2:
        return None
    
    # Simple peak detection
    peaks = []
    for i in range(1, len(right_side) - 1):
        if right_side[i] > right_side[i-1] and right_side[i] > right_side[i+1]:
            if right_side[i] > 0.1:  # Minimum height threshold
                peaks.append(i + center_idx + min_period)
    
    if len(peaks) == 0:
        return None
    
    # Use first prominent peak
    first_peak_idx = peaks[0]
    period = first_peak_idx - center_idx
    
    # Sanity check
    if period < min_period or period > max_period:
        return None
    
    return float(period)


def estimate_row_col_periods_autocorr(
    autocorr: np.ndarray,
) -> Tuple[Optional[float], Optional[float]]:
    """Estimate row and column periods from 2D autocorrelation.
    
    Args:
        autocorr: 2D autocorrelation map (zero-centered)
        
    Returns:
        (row_period, column_period) or (None, None) if estimation fails
    """
    h, w = autocorr.shape
    
    # Extract center row and column
    center_row = autocorr[h // 2, :]
    center_col = autocorr[:, w // 2]
    
    # Estimate periods
    row_period = estimate_period_from_autocorr_1d(center_row, w // 2)
    col_period = estimate_period_from_autocorr_1d(center_col, h // 2)
    
    return row_period, col_period
