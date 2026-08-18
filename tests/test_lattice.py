"""Unit tests for DRAM lattice construction."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dram_model.lattice_builder import build_dram_lattice, refine_lattice_with_reference, LatticeCandidate
from dram_model.pitch_estimator import estimate_dram_pitches


def test_pitch_estimation():
    """Test DRAM pitch estimation on synthetic DRAM image."""
    search_path = PROJECT_ROOT / "sample_0001_A_search.png"
    if not search_path.exists():
        print(f"Test image not found: {search_path}")
        return False
    
    image = np.asarray(Image.open(search_path).convert("L"), dtype=np.uint8)
    
    # Estimate pitches
    result = estimate_dram_pitches(image)
    
    print(f"Pitch Estimation Result:")
    print(f"  Row pitch: {result.row_pitch:.2f} px")
    print(f"  Column pitch: {result.column_pitch:.2f} px")
    print(f"  Orientation: {result.orientation:.2f} degrees")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Method: {result.method}")
    print(f"  Row variance: {result.row_pitch_variance:.4f}")
    print(f"  Column variance: {result.column_pitch_variance:.4f}")
    
    # Sanity checks
    assert result.row_pitch > 0, "Row pitch must be positive"
    assert result.column_pitch > 0, "Column pitch must be positive"
    assert 0 <= result.confidence <= 1, "Confidence must be in [0, 1]"
    assert result.row_pitch_variance >= 0, "Row variance must be non-negative"
    assert result.column_pitch_variance >= 0, "Column variance must be non-negative"
    
    print("PASS: Pitch estimation test passed")
    return True


def test_lattice_construction():
    """Test DRAM lattice construction."""
    # Create a simple test lattice
    image_shape = (512, 512)
    row_pitch = 28.0
    column_pitch = 28.0
    orientation = 0.0
    
    lattice = build_dram_lattice(
        image_shape=image_shape,
        row_pitch=row_pitch,
        column_pitch=column_pitch,
        orientation=orientation,
        margin=10,
    )
    
    print(f"\nLattice Construction Result:")
    print(f"  Row pitch: {lattice.row_pitch:.2f} px")
    print(f"  Column pitch: {lattice.column_pitch:.2f} px")
    print(f"  Orientation: {lattice.orientation:.2f} degrees")
    print(f"  Number of candidates: {len(lattice.candidates)}")
    print(f"  Origin: ({lattice.origin_x:.2f}, {lattice.origin_y:.2f})")
    
    # Sanity checks
    assert len(lattice.candidates) > 0, "Lattice should have candidates"
    assert lattice.row_pitch == row_pitch, "Row pitch mismatch"
    assert lattice.column_pitch == column_pitch, "Column pitch mismatch"
    
    # Check that all candidates are within bounds
    h, w = image_shape
    margin = 10
    for candidate in lattice.candidates:
        assert margin <= candidate.x < w - margin, f"Candidate x out of bounds: {candidate.x}"
        assert margin <= candidate.y < h - margin, f"Candidate y out of bounds: {candidate.y}"
        assert 0 <= candidate.phase_x < 1, f"Phase x out of range: {candidate.phase_x}"
        assert 0 <= candidate.phase_y < 1, f"Phase y out of range: {candidate.phase_y}"
    
    print("PASS: Lattice construction test passed")
    return True


def test_lattice_refinement():
    """Test lattice refinement with reference size."""
    image_shape = (512, 512)
    row_pitch = 28.0
    column_pitch = 28.0
    reference_size = (65, 65)
    
    lattice = build_dram_lattice(
        image_shape=image_shape,
        row_pitch=row_pitch,
        column_pitch=column_pitch,
        orientation=0.0,
        margin=10,
    )
    
    initial_count = len(lattice.candidates)
    
    refined_lattice = refine_lattice_with_reference(
        lattice=lattice,
        reference_size=reference_size,
        search_size=image_shape,
    )
    
    refined_count = len(refined_lattice.candidates)
    
    print(f"\nLattice Refinement Result:")
    print(f"  Initial candidates: {initial_count}")
    print(f"  Refined candidates: {refined_count}")
    print(f"  Removed: {initial_count - refined_count}")
    
    # Refined lattice should have fewer or equal candidates
    assert refined_count <= initial_count, "Refinement should not add candidates"
    
    # All refined candidates should accommodate the reference
    ref_h, ref_w = reference_size
    search_h, search_w = image_shape
    for candidate in refined_lattice.candidates:
        assert ref_w / 2 <= candidate.x <= search_w - ref_w / 2, "Candidate x too close to edge"
        assert ref_h / 2 <= candidate.y <= search_h - ref_h / 2, "Candidate y too close to edge"
    
    print("PASS: Lattice refinement test passed")
    return True


def test_lattice_candidate_operations():
    """Test lattice candidate query operations."""
    image_shape = (512, 512)
    row_pitch = 28.0
    column_pitch = 28.0
    
    lattice = build_dram_lattice(
        image_shape=image_shape,
        row_pitch=row_pitch,
        column_pitch=column_pitch,
        orientation=0.0,
        margin=10,
    )
    
    # Test get_candidate_at_position
    test_x, test_y = 256.0, 256.0
    closest = lattice.get_candidate_at_position(test_x, test_y, tolerance=5.0)
    
    print(f"\nCandidate Query Test:")
    print(f"  Query position: ({test_x}, {test_y})")
    if closest:
        print(f"  Closest candidate: ({closest.x:.2f}, {closest.y:.2f})")
        print(f"  Distance: {np.sqrt((closest.x - test_x)**2 + (closest.y - test_y)**2):.2f}")
    else:
        print("  No candidate found within tolerance")
    
    # Test get_candidates_in_region
    region_candidates = lattice.get_candidates_in_region(200, 300, 200, 300)
    print(f"  Candidates in region (200-300, 200-300): {len(region_candidates)}")
    
    print("PASS: Candidate operations test passed")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("DRAM Lattice Tests")
    print("=" * 60)
    
    all_passed = True
    
    try:
        all_passed &= test_pitch_estimation()
    except Exception as e:
        print(f"FAIL: Pitch estimation test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_lattice_construction()
    except Exception as e:
        print(f"FAIL: Lattice construction test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_lattice_refinement()
    except Exception as e:
        print(f"FAIL: Lattice refinement test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_lattice_candidate_operations()
    except Exception as e:
        print(f"FAIL: Candidate operations test failed: {e}")
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All lattice tests passed!")
    else:
        print("Some tests failed")
    print("=" * 60)
