"""Unit tests for candidate generation."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from candidate.candidate_generator import generate_candidates, CandidateGeneratorConfig, MatchCandidate
from candidate.top_k import select_top_k_candidates, TopKConfig
from dram_model.lattice_builder import build_dram_lattice
from dram_model.pitch_estimator import estimate_dram_pitches


def test_candidate_generation():
    """Test candidate generation from DRAM lattice."""
    search_path = PROJECT_ROOT / "sample_0001_A_search.png"
    if not search_path.exists():
        print(f"Test image not found: {search_path}")
        return False
    
    search_image = np.asarray(Image.open(search_path).convert("L"), dtype=np.uint8)
    
    # Estimate pitches and build lattice
    pitch_result = estimate_dram_pitches(search_image)
    lattice = build_dram_lattice(
        image_shape=search_image.shape,
        row_pitch=pitch_result.row_pitch,
        column_pitch=pitch_result.column_pitch,
        orientation=pitch_result.orientation,
        margin=10,
    )
    
    # Generate candidates
    config = CandidateGeneratorConfig(max_candidates=100, use_edge_density=True)
    candidates = generate_candidates(
        search_image=search_image,
        lattice=lattice,
        reference_size=(65, 65),
        config=config,
    )
    
    print(f"Candidate Generation Result:")
    print(f"  Lattice candidates: {len(lattice.candidates)}")
    print(f"  Generated candidates: {len(candidates)}")
    print(f"  Config max candidates: {config.max_candidates}")
    
    if candidates:
        print(f"  First candidate: ({candidates[0].x:.2f}, {candidates[0].y:.2f})")
        print(f"  Edge density range: {min(c.edge_density for c in candidates):.4f} - {max(c.edge_density for c in candidates):.4f}")
    
    # Sanity checks
    assert len(candidates) > 0, "Should generate at least some candidates"
    assert len(candidates) <= config.max_candidates, "Should respect max candidates"
    
    for candidate in candidates:
        assert candidate.edge_density >= 0, "Edge density should be non-negative"
        assert 0 <= candidate.phase_x < 1, "Phase x should be in [0, 1)"
        assert 0 <= candidate.phase_y < 1, "Phase y should be in [0, 1)"
    
    print("PASS: Candidate generation test passed")
    return True


def test_top_k_selection():
    """Test Top-K candidate selection."""
    # Create dummy candidates
    candidates = []
    for i in range(50):
        candidate = MatchCandidate(
            x=float(i * 10),
            y=float(i * 10),
            row_index=i,
            column_index=i,
            phase_x=0.5,
            phase_y=0.5,
            edge_density=np.random.rand(),
        )
        candidates.append(candidate)
    
    # Select top K
    config = TopKConfig(k=10, scoring_function="edge_density")
    top_k = select_top_k_candidates(candidates, config)
    
    print(f"\nTop-K Selection Result:")
    print(f"  Input candidates: {len(candidates)}")
    print(f"  Selected K: {len(top_k)}")
    print(f"  Config K: {config.k}")
    
    if top_k:
        print(f"  Best edge density: {top_k[0].edge_density:.4f}")
        print(f"  Worst edge density: {top_k[-1].edge_density:.4f}")
    
    # Sanity checks
    assert len(top_k) <= config.k, "Should not exceed K"
    assert len(top_k) <= len(candidates), "Should not exceed input candidates"
    
    # Check that candidates are sorted by score
    for i in range(len(top_k) - 1):
        assert top_k[i].edge_density >= top_k[i + 1].edge_density, "Should be sorted by score"
    
    print("PASS: Top-K selection test passed")
    return True


def test_phase_based_scoring():
    """Test phase-based candidate scoring."""
    # Create candidates with different phases
    candidates = []
    target_phase_x, target_phase_y = 0.5, 0.5
    
    for i in range(10):
        phase_x = (i / 10.0) % 1.0
        phase_y = (i / 10.0) % 1.0
        candidate = MatchCandidate(
            x=float(i * 10),
            y=float(i * 10),
            row_index=i,
            column_index=i,
            phase_x=phase_x,
            phase_y=phase_y,
            edge_density=0.5,  # Constant edge density
        )
        candidates.append(candidate)
    
    # Select using phase-based scoring
    config = TopKConfig(k=5, scoring_function="phase")
    top_k = select_top_k_candidates(
        candidates, config, target_phase_x, target_phase_y
    )
    
    print(f"\nPhase-Based Scoring Result:")
    print(f"  Target phase: ({target_phase_x:.2f}, {target_phase_y:.2f})")
    print(f"  Selected: {len(top_k)}")
    
    if top_k:
        print(f"  Best candidate phase: ({top_k[0].phase_x:.2f}, {top_k[0].phase_y:.2f})")
    
    # Sanity checks
    assert len(top_k) > 0, "Should select some candidates"
    
    print("PASS: Phase-based scoring test passed")
    return True


def test_combined_scoring():
    """Test combined edge density and phase scoring."""
    # Create candidates with varying edge density and phase
    candidates = []
    target_phase_x, target_phase_y = 0.5, 0.5
    
    for i in range(20):
        phase_x = (i / 20.0) % 1.0
        phase_y = (i / 20.0) % 1.0
        candidate = MatchCandidate(
            x=float(i * 10),
            y=float(i * 10),
            row_index=i,
            column_index=i,
            phase_x=phase_x,
            phase_y=phase_y,
            edge_density=np.random.rand(),
        )
        candidates.append(candidate)
    
    # Select using combined scoring
    config = TopKConfig(k=10, scoring_function="combined", edge_weight=0.7, phase_weight=0.3)
    top_k = select_top_k_candidates(
        candidates, config, target_phase_x, target_phase_y
    )
    
    print(f"\nCombined Scoring Result:")
    print(f"  Edge weight: {config.edge_weight}")
    print(f"  Phase weight: {config.phase_weight}")
    print(f"  Selected: {len(top_k)}")
    
    # Sanity checks
    assert len(top_k) > 0, "Should select some candidates"
    assert len(top_k) <= config.k, "Should respect K"
    
    print("PASS: Combined scoring test passed")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Candidate Generation Tests")
    print("=" * 60)
    
    all_passed = True
    
    try:
        all_passed &= test_candidate_generation()
    except Exception as e:
        print(f"FAIL: Candidate generation test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_top_k_selection()
    except Exception as e:
        print(f"FAIL: Top-K selection test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_phase_based_scoring()
    except Exception as e:
        print(f"FAIL: Phase-based scoring test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_combined_scoring()
    except Exception as e:
        print(f"FAIL: Combined scoring test failed: {e}")
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All candidate tests passed!")
    else:
        print("Some tests failed")
    print("=" * 60)
