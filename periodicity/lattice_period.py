"""Lattice period estimation combining FFT and autocorrelation."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .fft_analysis import analyze_periodicity, PeriodicityResult


def estimate_lattice_period(
    image: np.ndarray,
    use_autocorrelation_fallback: bool = True,
    fft_confidence_threshold: float = 0.5,
) -> PeriodicityResult:
    """Estimate DRAM lattice period using FFT with autocorrelation fallback.
    
    Args:
        image: Input image (grayscale or will be converted)
        use_autocorrelation_fallback: Whether to use autocorrelation if FFT confidence is low
        fft_confidence_threshold: Minimum FFT confidence to accept FFT result
        
    Returns:
        PeriodicityResult with estimated periods
    """
    # Try FFT first
    fft_result = analyze_periodicity(image)
    
    # If FFT confidence is high enough, return it
    if fft_result.confidence >= fft_confidence_threshold:
        return fft_result
    
    # Otherwise, try autocorrelation as fallback
    if use_autocorrelation_fallback:
        ac_result = _estimate_from_autocorrelation(image)
        
        # Combine results if autocorrelation succeeded
        if ac_result is not None:
            # Weighted combination based on confidence
            total_confidence = fft_result.confidence + ac_result.confidence
            if total_confidence > 0:
                weight_fft = fft_result.confidence / total_confidence
                weight_ac = ac_result.confidence / total_confidence
                
                combined_row = (weight_fft * fft_result.row_period + 
                               weight_ac * ac_result.row_period)
                combined_col = (weight_fft * fft_result.column_period + 
                               weight_ac * ac_result.column_period)
                combined_orientation = (weight_fft * fft_result.orientation + 
                                       weight_ac * ac_result.orientation)
                combined_confidence = min(0.95, fft_result.confidence + 0.2 * ac_result.confidence)
                
                return PeriodicityResult(
                    row_period=combined_row,
                    column_period=combined_col,
                    orientation=combined_orientation,
                    confidence=combined_confidence,
                    frequency_peaks=fft_result.frequency_peaks,
                    symmetric_pairs=fft_result.symmetric_pairs,
                    method="fft+autocorrelation",
                )
    
    return fft_result


def _estimate_from_autocorrelation(image: np.ndarray) -> Optional[PeriodicityResult]:
    """Estimate period using 2D autocorrelation.
    
    Args:
        image: Input image
        
    Returns:
        PeriodicityResult or None if estimation fails
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
    autocorr = autocorr / autocorr.max()
    
    h, w = autocorr.shape
    
    # Extract center row and column for 1D analysis
    center_row = autocorr[h // 2, :]
    center_col = autocorr[:, w // 2]
    
    # Find peaks in autocorrelation (excluding DC at center)
    def find_autocorr_peaks(signal, center_idx, min_distance=5):
        """Find peaks in 1D autocorrelation signal using simple method."""
        # Look at right side only (since autocorrelation is symmetric)
        right_side = signal[center_idx + min_distance:]
        
        if len(right_side) < min_distance * 2:
            return []
        
        # Simple peak detection: find local maxima
        peaks = []
        for i in range(1, len(right_side) - 1):
            if right_side[i] > right_side[i-1] and right_side[i] > right_side[i+1]:
                if right_side[i] > 0.1:  # Minimum height threshold
                    peaks.append(i + center_idx + min_distance)
        
        return peaks
    
    row_peaks = find_autocorr_peaks(center_row, w // 2)
    col_peaks = find_autocorr_peaks(center_col, h // 2)
    
    if not row_peaks or not col_peaks:
        return None
    
    # Estimate periods from first peaks
    row_period = float(row_peaks[0] - w // 2)
    col_period = float(col_peaks[0] - h // 2)
    
    # Sanity check
    if row_period < 3 or col_period < 3 or row_period > w // 3 or col_period > h // 3:
        return None
    
    # Use median of first few peaks for robustness
    if len(row_peaks) >= 3:
        row_periods = [p - w // 2 for p in row_peaks[:3]]
        row_period = float(np.median(row_periods))
    
    if len(col_peaks) >= 3:
        col_periods = [p - h // 2 for p in col_peaks[:3]]
        col_period = float(np.median(col_periods))
    
    return PeriodicityResult(
        row_period=row_period,
        column_period=col_period,
        orientation=0.0,
        confidence=0.7,
        frequency_peaks=[],
        symmetric_pairs=0,
        method="autocorrelation",
    )
