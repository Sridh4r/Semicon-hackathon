"""Unit tests for RCPF and fingerprinting."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fingerprint.row_column_phase import calculate_rcpf, RCPFResult, compare_rcpf, batch_calculate_rcpf
from fingerprint.dram_fingerprint import compute_dram_fingerprint, compare_dram_fingerprints
from fingerprint.neighborhood_fingerprint import compute_neighborhood_fingerprint, compare_neighborhood_fingerprints


def test_rcpf_calculation():
    """Test RCPF calculation for various positions."""
    row_pitch = 28.0
    column_pitch = 28.0
    
    # Test center position
    rcpf = calculate_rcpf(256.0, 256.0, row_pitch, column_pitch)
    
    print(f"RCPF Calculation Result:")
    print(f"  Position: (256.0, 256.0)")
    print(f"  Phase X: {rcpf.phase_x:.4f}")
    print(f"  Phase Y: {rcpf.phase_y:.4f}")
    print(f"  Row Phase: {rcpf.row_phase:.4f}")
    print(f"  Column Phase: {rcpf.column_phase:.4f}")
    
    # Sanity checks
    assert 0 <= rcpf.phase_x < 1, "Phase X should be in [0, 1)"
    assert 0 <= rcpf.phase_y < 1, "Phase Y should be in [0, 1)"
    assert rcpf.row_pitch == row_pitch, "Row pitch mismatch"
    assert rcpf.column_pitch == column_pitch, "Column pitch mismatch"
    
    # Test that different positions give different phases
    rcpf2 = calculate_rcpf(260.0, 260.0, row_pitch, column_pitch)
    assert rcpf.phase_x != rcpf2.phase_x or rcpf.phase_y != rcpf2.phase_y, \
        "Different positions should give different phases"
    
    print("PASS: RCPF calculation test passed")
    return True


def test_rcpf_comparison():
    """Test RCPF comparison function."""
    row_pitch = 28.0
    column_pitch = 28.0
    
    # Create two RCPF results
    rcpf1 = calculate_rcpf(256.0, 256.0, row_pitch, column_pitch)
    rcpf2 = calculate_rcpf(256.0, 256.0, row_pitch, column_pitch)  # Same position
    rcpf3 = calculate_rcpf(280.0, 280.0, row_pitch, column_pitch)  # Different position
    
    # Compare identical positions
    sim_identical = compare_rcpf(rcpf1, rcpf2)
    print(f"\nRCPF Comparison Result:")
    print(f"  Identical positions similarity: {sim_identical:.4f}")
    
    # Compare different positions
    sim_different = compare_rcpf(rcpf1, rcpf3)
    print(f"  Different positions similarity: {sim_different:.4f}")
    
    # Sanity checks
    assert sim_identical > 0.9, "Identical positions should have high similarity"
    assert sim_different < sim_identical, "Different positions should have lower similarity"
    assert 0 <= sim_identical <= 1, "Similarity should be in [0, 1]"
    assert 0 <= sim_different <= 1, "Similarity should be in [0, 1]"
    
    print("PASS: RCPF comparison test passed")
    return True


def test_batch_rcpf():
    """Test batch RCPF calculation."""
    row_pitch = 28.0
    column_pitch = 28.0
    
    positions = [(256.0, 256.0), (260.0, 260.0), (280.0, 280.0)]
    rcpfs = batch_calculate_rcpf(positions, row_pitch, column_pitch)
    
    print(f"\nBatch RCPF Result:")
    print(f"  Input positions: {len(positions)}")
    print(f"  Output RCPFs: {len(rcpfs)}")
    
    # Sanity checks
    assert len(rcpfs) == len(positions), "Output count should match input count"
    
    for rcpf in rcpfs:
        assert 0 <= rcpf.phase_x < 1, "Phase X should be in [0, 1)"
        assert 0 <= rcpf.phase_y < 1, "Phase Y should be in [0, 1)"
    
    print("PASS: Batch RCPF test passed")
    return True


def test_dram_fingerprint():
    """Test DRAM fingerprint computation."""
    search_path = PROJECT_ROOT / "sample_0001_A_search.png"
    if not search_path.exists():
        print(f"Test image not found: {search_path}")
        return False
    
    image = np.asarray(Image.open(search_path).convert("L"), dtype=np.uint8)
    
    # Compute fingerprint for a position
    fp = compute_dram_fingerprint(
        image=image,
        x=256.0,
        y=256.0,
        row_pitch=28.0,
        column_pitch=28.0,
        patch_size=32,
    )
    
    print(f"\nDRAM Fingerprint Result:")
    print(f"  RCPF: phase_x={fp.rcpf.phase_x:.4f}, phase_y={fp.rcpf.phase_y:.4f}")
    print(f"  Edge histogram shape: {fp.edge_histogram.shape}")
    print(f"  Intensity stats: mean={fp.intensity_stats['mean']:.2f}, std={fp.intensity_stats['std']:.2f}")
    print(f"  Texture features: contrast={fp.texture_features['contrast']:.4f}")
    
    # Sanity checks
    assert fp.edge_histogram.shape == (8,), "Edge histogram should have 8 bins"
    assert fp.intensity_stats["mean"] >= 0, "Mean should be non-negative"
    assert fp.texture_features["contrast"] >= 0, "Contrast should be non-negative"
    
    print("PASS: DRAM fingerprint test passed")
    return True


def test_neighborhood_fingerprint():
    """Test 3×3 neighborhood fingerprint."""
    search_path = PROJECT_ROOT / "sample_0001_A_search.png"
    if not search_path.exists():
        print(f"Test image not found: {search_path}")
        return False
    
    image = np.asarray(Image.open(search_path).convert("L"), dtype=np.uint8)
    
    # Compute neighborhood fingerprint
    nf = compute_neighborhood_fingerprint(
        image=image,
        x=256.0,
        y=256.0,
        row_pitch=28.0,
        column_pitch=28.0,
        neighborhood_size=3,
        patch_size=16,
    )
    
    print(f"\nNeighborhood Fingerprint Result:")
    print(f"  Center features: {nf.center_features}")
    print(f"  Number of neighbors: {len(nf.neighbor_features)}")
    print(f"  Relative positions: {nf.relative_positions}")
    print(f"  Neighborhood consistency: {nf.neighborhood_consistency:.4f}")
    
    # Sanity checks
    assert len(nf.neighbor_features) <= 8, "Should have at most 8 neighbors (3×3 minus center)"
    assert 0 <= nf.neighborhood_consistency <= 1, "Consistency should be in [0, 1]"
    
    print("PASS: Neighborhood fingerprint test passed")
    return True


def test_fingerprint_comparison():
    """Test fingerprint comparison functions."""
    search_path = PROJECT_ROOT / "sample_0001_A_search.png"
    if not search_path.exists():
        print(f"Test image not found: {search_path}")
        return False
    
    image = np.asarray(Image.open(search_path).convert("L"), dtype=np.uint8)
    
    # Compute fingerprints for two positions
    fp1 = compute_dram_fingerprint(image, 256.0, 256.0, 28.0, 28.0)
    fp2 = compute_dram_fingerprint(image, 256.0, 256.0, 28.0, 28.0)  # Same position
    fp3 = compute_dram_fingerprint(image, 280.0, 280.0, 28.0, 28.0)  # Different position
    
    # Compare DRAM fingerprints
    sim_identical = compare_dram_fingerprints(fp1, fp2)
    sim_different = compare_dram_fingerprints(fp1, fp3)
    
    print(f"\nFingerprint Comparison Result:")
    print(f"  Identical positions similarity: {sim_identical:.4f}")
    print(f"  Different positions similarity: {sim_different:.4f}")
    
    # Sanity checks
    assert sim_identical > sim_different, "Identical positions should be more similar"
    assert 0 <= sim_identical <= 1, "Similarity should be in [0, 1]"
    assert 0 <= sim_different <= 1, "Similarity should be in [0, 1]"
    
    print("PASS: Fingerprint comparison test passed")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("RCPF and Fingerprinting Tests")
    print("=" * 60)
    
    all_passed = True
    
    try:
        all_passed &= test_rcpf_calculation()
    except Exception as e:
        print(f"FAIL: RCPF calculation test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_rcpf_comparison()
    except Exception as e:
        print(f"FAIL: RCPF comparison test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_batch_rcpf()
    except Exception as e:
        print(f"FAIL: Batch RCPF test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_dram_fingerprint()
    except Exception as e:
        print(f"FAIL: DRAM fingerprint test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_neighborhood_fingerprint()
    except Exception as e:
        print(f"FAIL: Neighborhood fingerprint test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_fingerprint_comparison()
    except Exception as e:
        print(f"FAIL: Fingerprint comparison test failed: {e}")
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All fingerprinting tests passed!")
    else:
        print("Some tests failed")
    print("=" * 60)
