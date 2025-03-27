import pytest
from src.sort_orbits import sort_orbits

def test_sort_orbits_basic():
    """Test basic sorting functionality."""
    orbits = [(1,2), (0,), (1,2,3), (0,1)]
    expected = [(0,), (0,1), (1,2), (1,2,3)]
    assert sort_orbits(orbits) == expected, f"Expected {expected}, got {sort_orbits(orbits)}"

def test_sort_orbits_empty():
    """Test with empty list."""
    assert sort_orbits([]) == [], "Empty list should return empty list"

def test_sort_orbits_single_element():
    """Test with single element."""
    orbits = [(1,)]
    assert sort_orbits(orbits) == orbits, "Single element should remain unchanged"

def test_sort_orbits_same_length():
    """Test sorting when all orbits have same length."""
    orbits = [(2,1), (1,2), (0,3)]
    expected = [(0,3), (1,2), (2,1)]
    assert sort_orbits(orbits) == expected, f"Expected {expected}, got {sort_orbits(orbits)}"

def test_sort_orbits_duplicates():
    """Test handling of duplicate orbits."""
    orbits = [(1,2), (1,2), (0,)]
    expected = [(0,), (1,2), (1,2)]
    assert sort_orbits(orbits) == expected, f"Expected {expected}, got {sort_orbits(orbits)}" 