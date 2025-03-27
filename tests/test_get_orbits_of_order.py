import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from src.get_orbits_of_order import get_orbits_of_order

def test_get_orbits_of_order_order_1():
    """Test with order 1 from sequence length 3."""
    orbits = get_orbits_of_order(3, 1)
    expected = [(0,), (1,), (2,)]
    assert orbits == expected, f"Expected {expected}, got {orbits}"

def test_get_orbits_of_order_order_2():
    """Test with order 2 from sequence length 3."""
    orbits = get_orbits_of_order(3, 2)
    expected = [(0, 1), (0, 2), (1, 2)]
    assert orbits == expected, f"Expected {expected}, got {orbits}"

def test_get_orbits_of_order_full_positions():
    """Test with order equal to sequence length (all positions)."""
    orbits = get_orbits_of_order(3, 3)
    expected = [(0, 1, 2)]
    assert orbits == expected, f"Expected {expected}, got {orbits}"

def test_get_orbits_of_order_empty_selection():
    """Test with order 0 (empty selection)."""
    orbits = get_orbits_of_order(3, 0)
    expected = [()]  # One combination with zero elements
    assert orbits == expected, f"Expected {expected}, got {orbits}"

def test_get_orbits_of_order_larger_sequence():
    """Test with order 2 from sequence length 5."""
    orbits = get_orbits_of_order(5, 2)
    expected = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    assert orbits == expected, f"Expected {expected}, got {orbits}"

def test_get_orbits_of_order_count_check():
    """Check count for order 3 from sequence length 6."""
    # Number of combinations = C(6,3) = 6!/(3!*(6-3)!) = 20
    orbits = get_orbits_of_order(6, 3)
    assert len(orbits) == 20, f"Expected 20 combinations, got {len(orbits)}"
    
    # Verify each orbit contains unique positions
    for orbit in orbits:
        assert len(set(orbit)) == len(orbit), f"Orbit {orbit} contains duplicate positions"

def test_get_orbits_of_order_validation():
    """Test validation logic."""
    # Test order > sequence length
    with pytest.raises(ValueError, match="Order .* cannot exceed sequence length"):
        get_orbits_of_order(5, 6)
    
    # Test negative order
    with pytest.raises(ValueError, match="Order must be non-negative"):
        get_orbits_of_order(5, -1) 