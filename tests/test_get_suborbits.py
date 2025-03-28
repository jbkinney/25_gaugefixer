import pytest
from src._get_suborbits import _get_suborbits

def test_get_suborbits_single_orbit():
    """Test with a single orbit."""
    suborbits = _get_suborbits([(0, 1, 2)])
    expected = [(), (0,), (0, 1), (0, 1, 2), (0, 2), (1,), (1, 2), (2,)]
    assert set(suborbits) == set(expected), f"Expected {expected}, got {suborbits}"

def test_get_suborbits_multiple_orbits():
    """Test with multiple orbits."""
    suborbits = _get_suborbits([(0, 1, 2), (1, 2, 3)])
    expected = [(), (0,), (0, 1), (0, 1, 2), (0, 2), (1,), (1, 2), (1, 2, 3), (1, 3), (2,), (2, 3), (3,)]
    assert set(suborbits) == set(expected), f"Expected {expected}, got {suborbits}"

def test_get_suborbits_empty_orbit():
    """Test with an empty orbit."""
    suborbits = _get_suborbits([()])
    expected = [()]
    assert suborbits == expected, f"Expected {expected}, got {suborbits}"

def test_get_suborbits_single_position():
    """Test with single-position orbits."""
    suborbits = _get_suborbits([(0,), (1,), (2,)])
    expected = [(), (0,), (1,), (2,)]
    assert set(suborbits) == set(expected), f"Expected {expected}, got {suborbits}"

def test_get_suborbits_no_duplicates():
    """Test that no duplicate suborbits are returned."""
    suborbits = _get_suborbits([(0, 1), (1, 0)])  # Same positions, different order
    expected = [(), (0,), (0, 1), (1,)]
    assert set(suborbits) == set(expected), f"Expected {expected}, got {suborbits}" 