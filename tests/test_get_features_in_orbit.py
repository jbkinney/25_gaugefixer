import pytest
from src._get_features_in_orbit import _get_features_in_orbit

def test_get_features_in_orbit_single_position():
    """Test with a single position, binary alphabet."""
    seqs = _get_features_in_orbit(['A', 'B'], [1], 3)
    expected = ['*A*', '*B*']
    assert set(seqs) == set(expected), f"Expected {expected}, got {seqs}"

def test_get_features_in_orbit_two_positions():
    """Test with two positions, binary alphabet."""
    seqs = _get_features_in_orbit(['A', 'B'], [0, 2], 3)
    expected = ['A*A', 'A*B', 'B*A', 'B*B']
    assert set(seqs) == set(expected), f"Expected {expected}, got {seqs}"

def test_get_features_in_orbit_three_positions():
    """Test with three positions, different alphabet."""
    seqs = _get_features_in_orbit(['C', 'D'], [0, 1, 2], 3)
    # Check the count - should be 2³ = 8
    assert len(seqs) == 8, f"Expected 8 sequences, got {len(seqs)}"
    # Check specific sequences
    assert 'CCC' in seqs, "Sequence with all C characters should be in results"
    assert 'DDD' in seqs, "Sequence with all D characters should be in results"

def test_get_features_in_orbit_validation():
    """Test validation logic."""
    # Test position out of range
    with pytest.raises(ValueError, match="Positions must be within range"):
        _get_features_in_orbit(['A', 'B'], [3], 3)
    
    # Test empty alphabet
    with pytest.raises(ValueError, match="Alphabet must not be empty"):
        _get_features_in_orbit([], [0], 3)

def test_get_features_in_orbit_custom_wildcard():
    """Test with a custom wildcard character."""
    seqs = _get_features_in_orbit(['A', 'B'], [1], 3, wildcard_char='?')
    expected = ['?A?', '?B?']
    assert set(seqs) == set(expected), f"Expected {expected}, got {seqs}" 