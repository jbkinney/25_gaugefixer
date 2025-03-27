import numpy as np
from typing import List, Tuple
from itertools import combinations
from src.get_orbits_of_order import get_orbits_of_order
from src.sort_orbits import sort_orbits

def get_adjacent_orbits_of_order(
    seq_length: int,
    order: int
) -> List[Tuple[int, ...]]:
    """
    Generate all combinations of sequence positions where the min and max positions 
    differ by at most the specified order.
    
    This function returns all possible combinations where the span between the minimum
    and maximum position is less than or equal to the order.
    
    Args:
        seq_length (int): The total length of the sequence
        order (int): The maximum difference between min and max positions in each combination
        
    Returns:
        List[Tuple[int, ...]]: List of tuples, where each tuple contains positions 
                              with max-min <= order
    
    Examples:
        >>> get_adjacent_orbits_of_order(5, 3)
        [(0, 1, 2), (1, 2, 3), (2, 3, 4)]
    """
    # Validate inputs
    if order < 1:
        raise ValueError("Order must be greater or equal to 1")
    
    if order > seq_length:
        raise ValueError(f"Order ({order}) cannot exceed sequence length ({seq_length})")
    
    # Get all possible combinations of 2 positions
    orbits = [tuple(range(i, i+order)) for i in range(seq_length-order+1)] 
    
    return orbits
    