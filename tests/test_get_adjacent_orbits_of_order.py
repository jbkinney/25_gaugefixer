import pytest
from src._get_adjacent_orbits_of_order import _get_adjacent_orbits_of_order

def test_get_adjacent_orbits_of_order_order_2():
    """Test with order 2, sequence length 5."""
    orbits = _get_adjacent_orbits_of_order(5, 2)
    expected = [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert set(orbits) == set(expected), f"Expected {expected}, got {orbits}"

def test_get_adjacent_orbits_of_order_order_3():
    """Test with order 3, sequence length 5."""
    orbits = _get_adjacent_orbits_of_order(5, 3)
    expected = [(0, 1, 2), (1, 2, 3), (2, 3, 4)]
    assert set(orbits) == set(expected), f"Expected {expected}, got {orbits}"

def test_get_adjacent_orbits_of_order_order_1():
    """Test with order 1, sequence length 3."""
    orbits = _get_adjacent_orbits_of_order(3, 1)
    expected = [(0,), (1,), (2,)]
    assert set(orbits) == set(expected), f"Expected {expected}, got {orbits}"

def test_get_adjacent_orbits_of_order_order_equals_length():
    """Test with order equal to sequence length."""
    orbits = _get_adjacent_orbits_of_order(3, 3)
    expected = [(0, 1, 2)]
    assert set(orbits) == set(expected), f"Expected {expected}, got {orbits}"

def test_get_adjacent_orbits_of_order_validation():
    """Test validation logic."""
    # Test negative order
    with pytest.raises(ValueError, match="Order must be greater or equal to 1"):
        _get_adjacent_orbits_of_order(5, 0)
    
    # Test order > sequence length
    with pytest.raises(ValueError, match="Order .* cannot exceed sequence length"):
        _get_adjacent_orbits_of_order(5, 6)

def test_get_adjacent_orbits_of_order_count_check():
    """Check count for order 2 from sequence length 5."""
    orbits = _get_adjacent_orbits_of_order(5, 2)
    # For each starting position (0 to 3), we get C(2,2) = 1 combination
    # Total: 4 starting positions * 1 combination = 4 combinations
    assert len(orbits) == 4, f"Expected 4 combinations, got {len(orbits)}"
    
    # Verify each orbit contains unique positions
    for orbit in orbits:
        assert len(set(orbit)) == len(orbit), f"Orbit {orbit} contains duplicate positions"
