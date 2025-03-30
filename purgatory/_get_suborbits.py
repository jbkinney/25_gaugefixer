from typing import List, Tuple
from itertools import combinations
from purgatory._sort_orbits import _sort_orbits

def _get_suborbits(orbits: List[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
    """
    Get all possible subsets of positions from a list of orbits.
    
    For each orbit (tuple of positions), generates all possible subsets of those positions.
    Returns a sorted list of unique suborbits.
    
    Args:
        orbits (List[Tuple[int, ...]]): List of orbits, where each orbit is a tuple of positions
        
    Returns:
        List[Tuple[int, ...]]: Sorted list of all unique suborbits
        
    Examples:
        >>> _get_suborbits([(0,1,2), (1,2,3)])
        [(), (0,), (0,1), (0,1,2), (0,2), (1,), (1,2), (1,2,3), (1,3), (2,), (2,3), (3,)]
    """
    # Sort each orbit
    orbits = [tuple(sorted(orbit)) for orbit in orbits]
    
    # Remove duplicate orbits
    orbits = list(set(orbits))
    
    # Use set to avoid duplicates
    suborbits = set()
    
    # For each orbit
    for orbit in orbits:
        # Generate all possible subset sizes from 1 to len(orbit)
        for size in range(0, len(orbit) + 1):
            # Get all combinations of that size
            suborbits.update(combinations(orbit, size))
            
    # Convert to sorted list and return
    return _sort_orbits(list(suborbits)) 