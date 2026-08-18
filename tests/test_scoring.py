"""Unit tests for verification and scoring."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from verification.topology_score import compute_topology_score
from verification.final_score import compute_final_score, ScoreWeights, compute_geometry_score
from verification.confidence import compute_confidence, is_high_confidence


def test_topology_score():
    """Test DRAM topology scoring."""
    search_path = PROJECT_ROOT / "sample_0001_A_search.png"
    if not search_path.exists():
        print(f"Test image not found: {search_path}")
        return False
    
    image = np.asarray(Image.open(search_path).convert("L"), dtype=np.uint8)
    
    # Compute topology score for a position
    result = compute_topology_score(
        image=image,
        x=256.0,
        y=256.0,
        row_pitch=28.0,
        column_pitch=28.0,
        neighborhood_size=3,
    )
    
    print(f"Topology Score Result:")
    print(f"  Horizontal consistency: {result.horizontal_consistency:.4f}")
    print(f"  Vertical consistency: {result.vertical_consistency:.4f}")
    print(f"  Intersection consistency: {result.intersection_consistency:.4f}")
    print(f"  Pitch consistency: {result.pitch_consistency:.4f}")
    print(f"  Neighbor alignment: {result.neighbor_alignment:.4f}")
    print(f"  Overall score: {result.overall_score:.4f}")
    
    # Sanity checks
    assert 0 <= result.overall_score <= 1, "Overall score should be in [0, 1]"
    assert 0 <= result.horizontal_consistency <= 1, "Horizontal consistency should be in [0, 1]"
    assert 0 <= result.vertical_consistency <= 1, "Vertical consistency should be in [0, 1]"
    
    print("PASS: Topology score test passed")
    return True


def test_final_score():
    """Test final scoring computation."""
    # Create sample scores
    appearance_score = 0.85
    rcpf_score = 0.92
    neighborhood_score = 0.78
    topology_score = 0.88
    geometry_score = 0.70
    
    # Compute final score with default weights
    result = compute_final_score(
        appearance_score=appearance_score,
        rcpf_score=rcpf_score,
        neighborhood_score=neighborhood_score,
        topology_score=topology_score,
        geometry_score=geometry_score,
    )
    
    print(f"\nFinal Score Result:")
    print(f"  Final score: {result.final_score:.4f}")
    print(f"  Appearance: {result.appearance_score:.4f}")
    print(f"  RCPF: {result.rcpf_score:.4f}")
    print(f"  Neighborhood: {result.neighborhood_score:.4f}")
    print(f"  Topology: {result.topology_score:.4f}")
    print(f"  Geometry: {result.geometry_score:.4f}")
    
    # Sanity checks
    assert 0 <= result.final_score <= 1, "Final score should be in [0, 1]"
    assert result.final_score > 0.7, "Should be reasonably high with good component scores"
    
    # Test with custom weights
    custom_weights = ScoreWeights(appearance=0.5, rcpf=0.3, neighborhood=0.1, topology=0.05, geometry=0.05)
    custom_result = compute_final_score(
        appearance_score=appearance_score,
        rcpf_score=rcpf_score,
        neighborhood_score=neighborhood_score,
        topology_score=topology_score,
        geometry_score=geometry_score,
        weights=custom_weights,
    )
    
    print(f"  Custom weights final score: {custom_result.final_score:.4f}")
    
    print("PASS: Final score test passed")
    return True


def test_geometry_score():
    """Test geometry score computation."""
    # Test center position
    center_score = compute_geometry_score(256.0, 256.0, 512, 512, preferred_center=True)
    
    # Test corner position
    corner_score = compute_geometry_score(50.0, 50.0, 512, 512, preferred_center=True)
    
    # Test neutral (no center preference)
    neutral_score = compute_geometry_score(256.0, 256.0, 512, 512, preferred_center=False)
    
    print(f"\nGeometry Score Result:")
    print(f"  Center position score: {center_score:.4f}")
    print(f"  Corner position score: {corner_score:.4f}")
    print(f"  Neutral score: {neutral_score:.4f}")
    
    # Sanity checks
    assert center_score > corner_score, "Center should have higher score than corner"
    assert neutral_score == 0.5, "Neutral score should be 0.5"
    assert 0 <= center_score <= 1, "Center score should be in [0, 1]"
    assert 0 <= corner_score <= 1, "Corner score should be in [0, 1]"
    
    print("PASS: Geometry score test passed")
    return True


def test_confidence():
    """Test confidence estimation."""
    # High confidence case (clear winner)
    high_conf_result = compute_confidence(
        best_score=0.99,
        all_scores=[0.99, 0.50, 0.40, 0.30],
        rcpf_consistency=0.95,
        neighborhood_consistency=0.90,
        topology_consistency=0.95,
    )
    
    # Low confidence case (ambiguous)
    low_conf_result = compute_confidence(
        best_score=0.92,
        all_scores=[0.92, 0.91, 0.90, 0.89],
        rcpf_consistency=0.7,
        neighborhood_consistency=0.65,
        topology_consistency=0.68,
    )
    
    print(f"\nConfidence Result:")
    print(f"  High confidence case: {high_conf_result.overall_confidence:.4f}")
    print(f"  Low confidence case: {low_conf_result.overall_confidence:.4f}")
    print(f"  High confidence gap: {high_conf_result.score_gap:.4f}")
    print(f"  Low confidence gap: {low_conf_result.score_gap:.4f}")
    
    # Sanity checks
    assert high_conf_result.overall_confidence > low_conf_result.overall_confidence, \
        "High confidence case should have higher overall confidence"
    assert high_conf_result.score_gap > low_conf_result.score_gap, \
        "High confidence case should have larger score gap"
    assert 0 <= high_conf_result.overall_confidence <= 1, "Confidence should be in [0, 1]"
    assert 0 <= low_conf_result.overall_confidence <= 1, "Confidence should be in [0, 1]"
    
    # Test confidence classification
    assert is_high_confidence(high_conf_result), "Should be high confidence"
    assert not is_high_confidence(low_conf_result), "Should not be high confidence"
    assert is_ambiguous_confidence(low_conf_result), "Should be ambiguous"
    assert not is_ambiguous_confidence(high_conf_result), "Should not be ambiguous"
    
    print("PASS: Confidence test passed")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Verification and Scoring Tests")
    print("=" * 60)
    
    all_passed = True
    
    try:
        all_passed &= test_topology_score()
    except Exception as e:
        print(f"FAIL: Topology score test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_final_score()
    except Exception as e:
        print(f"FAIL: Final score test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_geometry_score()
    except Exception as e:
        print(f"FAIL: Geometry score test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_confidence()
    except Exception as e:
        print(f"FAIL: Confidence test failed: {e}")
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All verification tests passed!")
    else:
        print("Some tests failed")
    print("=" * 60)
