"""Robust DRAM pitch estimation using multiple methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from periodicity.fft_analysis import PeriodicityResult
from periodicity.autocorrelation import estimate_row_col_periods_autocorr, compute_autocorrelation


@dataclass(frozen=True)
class PitchEstimationResult:
    """Result of DRAM pitch estimation."""
    
    row_pitch: float
    column_pitch: float
    orientation: float
    confidence: float
    method: str
    row_pitch_variance: float
    column_pitch_variance: float


def estimate_dram_pitches(
    image: np.ndarray,
    fft_result: Optional[PeriodicityResult] = None,
    use_robust_statistics: bool = True,
    outlier_threshold: float = 2.0,
) -> PitchEstimationResult:
    """Estimate DRAM row and column pitches using robust statistics.
    
    Args:
        image: Input image (grayscale or will be converted)
        fft_result: Optional FFT periodicity result to use as starting point
        use_robust_statistics: Whether to apply robust statistical methods
        outlier_threshold: Standard deviation threshold for outlier rejection
        
    Returns:
        PitchEstimationResult with estimated pitches and confidence
    """
    from periodicity.lattice_period import estimate_lattice_period
    
    # Get FFT-based estimate if not provided
    if fft_result is None:
        fft_result = estimate_lattice_period(image)
    
    # Apply robust statistics to refine estimates
    if use_robust_statistics:
        return _refine_with_robust_statistics(image, fft_result, outlier_threshold)
    else:
        return PitchEstimationResult(
            row_pitch=fft_result.row_period,
            column_pitch=fft_result.column_period,
            orientation=fft_result.orientation,
            confidence=fft_result.confidence,
            method="fft_basic",
            row_pitch_variance=0.0,
            column_pitch_variance=0.0,
        )


def _refine_with_robust_statistics(
    image: np.ndarray,
    fft_result: PeriodicityResult,
    outlier_threshold: float,
) -> PitchEstimationResult:
    """Refine pitch estimates using robust statistical methods."""
    # Collect multiple estimates
    estimates = []
    
    # 1. FFT-based estimate
    estimates.append((fft_result.row_period, fft_result.column_period))
    
    # 2. Autocorrelation-based estimate
    try:
        from periodicity.autocorrelation import compute_autocorrelation, estimate_row_col_periods_autocorr
        autocorr = compute_autocorrelation(image)
        row_ac, col_ac = estimate_row_col_periods_autocorr(autocorr)
        if row_ac is not None and col_ac is not None:
            estimates.append((row_ac, col_ac))
    except Exception:
        pass  # Autocorrelation may fail for some images
    
    # 3. Spatial domain estimate (using peak detection in image gradients)
    try:
        row_spatial, col_spatial = _estimate_from_spatial_domain(image)
        if row_spatial is not None and col_spatial is not None:
            estimates.append((row_spatial, col_spatial))
    except Exception:
        pass  # Spatial estimation may fail
    
    if not estimates:
        # Fallback to FFT result
        return PitchEstimationResult(
            row_pitch=fft_result.row_period,
            column_pitch=fft_result.column_period,
            orientation=fft_result.orientation,
            confidence=fft_result.confidence * 0.8,
            method="fft_fallback",
            row_pitch_variance=0.0,
            column_pitch_variance=0.0,
        )
    
    # Extract row and column estimates
    row_estimates = np.array([e[0] for e in estimates])
    col_estimates = np.array([e[1] for e in estimates])
    
    # Remove outliers using standard deviation
    if len(row_estimates) > 2:
        row_mean = np.mean(row_estimates)
        row_std = np.std(row_estimates)
        row_mask = np.abs(row_estimates - row_mean) <= outlier_threshold * row_std
        row_estimates = row_estimates[row_mask]
    
    if len(col_estimates) > 2:
        col_mean = np.mean(col_estimates)
        col_std = np.std(col_estimates)
        col_mask = np.abs(col_estimates - col_mean) <= outlier_threshold * col_std
        col_estimates = col_estimates[col_mask]
    
    # Use median for final estimate (robust to outliers)
    final_row_pitch = float(np.median(row_estimates))
    final_col_pitch = float(np.median(col_estimates))
    
    # Calculate variance as confidence measure
    row_variance = float(np.var(row_estimates)) if len(row_estimates) > 1 else 0.0
    col_variance = float(np.var(col_estimates)) if len(col_estimates) > 1 else 0.0
    
    # Combine variances for overall confidence
    combined_variance = (row_variance + col_variance) / 2
    robustness_bonus = 0.1 * len(estimates)  # More methods = higher confidence
    confidence = min(0.95, fft_result.confidence + robustness_bonus - combined_variance * 0.01)
    confidence = max(0.1, confidence)  # Ensure minimum confidence
    
    return PitchEstimationResult(
        row_pitch=final_row_pitch,
        column_pitch=final_col_pitch,
        orientation=fft_result.orientation,
        confidence=confidence,
        method="robust_combined",
        row_pitch_variance=row_variance,
        column_pitch_variance=col_variance,
    )


def _estimate_from_spatial_domain(image: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """Estimate pitches from spatial domain using gradient peaks."""
    import cv2
    
    # Convert to grayscale if needed
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    
    # Compute gradients
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute gradient magnitude
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Project gradients onto axes
    row_profile = np.mean(grad_mag, axis=1)  # Vertical profile
    col_profile = np.mean(grad_mag, axis=0)  # Horizontal profile
    
    # Find peaks in profiles
    def find_profile_peaks(profile, min_distance=5):
        """Find peaks in 1D profile."""
        peaks = []
        for i in range(min_distance, len(profile) - min_distance):
            if profile[i] > profile[i-1] and profile[i] > profile[i+1]:
                if profile[i] > np.mean(profile) * 0.5:  # Above mean threshold
                    peaks.append(i)
        return peaks
    
    row_peaks = find_profile_peaks(row_profile)
    col_peaks = find_profile_peaks(col_profile)
    
    if len(row_peaks) < 2 or len(col_peaks) < 2:
        return None, None
    
    # Estimate pitch from peak spacing
    row_spacings = [row_peaks[i+1] - row_peaks[i] for i in range(len(row_peaks)-1)]
    col_spacings = [col_peaks[i+1] - col_peaks[i] for i in range(len(col_peaks)-1)]
    
    row_pitch = float(np.median(row_spacings))
    col_pitch = float(np.median(col_spacings))
    
    return row_pitch, col_pitch
