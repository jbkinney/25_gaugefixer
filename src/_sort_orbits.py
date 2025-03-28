from typing import List, Tuple

def _sort_orbits(orbits: List[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
    """
    Sort orbits first by length, then by their contents.
    
    Args:
        orbits (List[Tuple[int, ...]]): List of orbits to sort
        
    Returns:
        List[Tuple[int, ...]]: Sorted list of orbits
        
    Examples:
        >>> _sort_orbits([(1,2), (0,), (1,2,3), (0,1)])
        [(0,), (1,), (0,1), (1,2), (1,2,3)]
    """
    return sorted(orbits, key=lambda x: (len(x), x)) 