"""Unit tests for periodicity analysis."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from periodicity.fft_analysis import analyze_periodicity
from periodicity.lattice_period import estimate_lattice_period


def test_fft_on_synthetic_dram():
    """Test FFT analysis on synthetic DRAM image."""
    # Load a synthetic DRAM image
    search_path = PROJECT_ROOT / "sample_0001_A_search.png"
    if not search_path.exists():
        print(f"Test image not found: {search_path}")
        return False
    
    image = np.asarray(Image.open(search_path).convert("L"), dtype=np.uint8)
    
    # Run FFT analysis
    result = analyze_periodicity(image)
    
    print(f"FFT Analysis Result:")
    print(f"  Row period: {result.row_period:.2f} px")
    print(f"  Column period: {result.column_period:.2f} px")
    print(f"  Orientation: {result.orientation:.2f} degrees")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Frequency peaks: {len(result.frequency_peaks)}")
    print(f"  Symmetric pairs: {result.symmetric_pairs}")
    
    # Basic sanity checks
    assert result.row_period > 0, "Row period must be positive"
    assert result.column_period > 0, "Column period must be positive"
    assert 0 <= result.confidence <= 1, "Confidence must be in [0, 1]"
    assert -90 <= result.orientation <= 90, "Orientation should be reasonable"
    
    # Check that periods are within reasonable range for the image
    h, w = image.shape
    assert result.row_period < h / 2, "Row period too large"
    assert result.column_period < w / 2, "Column period too large"
    
    print("PASS: FFT analysis test passed")
    return True


def test_lattice_period_estimation():
    """Test lattice period estimation with fallback."""
    search_path = PROJECT_ROOT / "sample_0001_A_search.png"
    if not search_path.exists():
        print(f"Test image not found: {search_path}")
        return False
    
    image = np.asarray(Image.open(search_path).convert("L"), dtype=np.uint8)
    
    # Run lattice period estimation
    result = estimate_lattice_period(image)
    
    print(f"\nLattice Period Estimation:")
    print(f"  Method: {result.method}")
    print(f"  Row period: {result.row_period:.2f} px")
    print(f"  Column period: {result.column_period:.2f} px")
    print(f"  Orientation: {result.orientation:.2f} degrees")
    print(f"  Confidence: {result.confidence:.2f}")
    
    # Sanity checks
    assert result.row_period > 0, "Row period must be positive"
    assert result.column_period > 0, "Column period must be positive"
    assert 0 <= result.confidence <= 1, "Confidence must be in [0, 1]"
    
    print("PASS: Lattice period estimation test passed")
    return True


def test_periodicity_on_noise_variants():
    """Test periodicity detection on different noise variants."""
    experiments = ["A", "B", "C", "D", "E"]
    
    for exp in experiments:
        search_path = PROJECT_ROOT / f"sample_0001_{exp}_search.png"
        if not search_path.exists():
            continue
        
        image = np.asarray(Image.open(search_path).convert("L"), dtype=np.uint8)
        result = estimate_lattice_period(image)
        
        print(f"\nExperiment {exp}:")
        print(f"  Row period: {result.row_period:.2f} px")
        print(f"  Column period: {result.column_period:.2f} px")
        print(f"  Confidence: {result.confidence:.2f}")
        
        # Periodicity should still be detectable even with noise
        assert result.confidence > 0.2, f"Confidence too low for experiment {exp}"
    
    print("PASS: Periodicity on noise variants test passed")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Periodicity Analysis Tests")
    print("=" * 60)
    
    all_passed = True
    
    try:
        all_passed &= test_fft_on_synthetic_dram()
    except Exception as e:
        print(f"FAIL: FFT test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_lattice_period_estimation()
    except Exception as e:
        print(f"FAIL: Lattice period test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_periodicity_on_noise_variants()
    except Exception as e:
        print(f"FAIL: Noise variants test failed: {e}")
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All periodicity tests passed!")
    else:
        print("Some tests failed")
    print("=" * 60)
