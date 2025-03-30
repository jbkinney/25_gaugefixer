import numpy as np
import pandas as pd
from typing import List, Tuple
from itertools import combinations
from purgatory._sort_orbits import _sort_orbits
def _get_orbits_of_order(
    seq_length: int,
    order: int
) -> List[Tuple[int, ...]]:
    """
    Generate all combinations of sequence positions of a given order.
    
    This function returns all possible combinations of 'order' positions 
    from a sequence of length 'seq_length'.
    
    Args:
        seq_length (int): The total length of the sequence
        order (int): The number of positions to select in each combination
        
    Returns:
        List[Tuple[int, ...]]: List of tuples, where each tuple contains 'order' 
                              distinct positions (0-indexed)
    
    Examples:
        >>> _get_orbits_of_order(3, 1)
        [(0,), (1,), (2,)]
        
        >>> _get_orbits_of_order(4, 2)
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    """
    # Validate inputs
    if order < 0:
        raise ValueError("Order must be non-negative")
    
    if order > seq_length:
        raise ValueError(f"Order ({order}) cannot exceed sequence length ({seq_length})")
    
    # Generate all combinations of 'order' positions from range(seq_length)
    orbits = list(combinations(range(seq_length), order))
    
    return _sort_orbits(orbits)