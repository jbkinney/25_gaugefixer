import numpy as np
import pandas as pd
from typing import List
from itertools import chain

from src._get_orbits_of_order import _get_orbits_of_order
from src._get_features_in_orbit import _get_features_in_orbit

def get_features_of_order(
    seq_length: int,
    order: int,
    alphabet: List[str],
    wildcard_char: str = '*'
) -> List[str]:
    """
    Generate all features of a given order.
    
    This function generates all features where exactly 'order' positions
    have characters from the alphabet, and all other positions have wildcards.
    It first finds all possible combinations of 'order' positions, then generates
    all possible features for each combination.
    
    Args:
        seq_length (int): The total length of the sequence
        order (int): The number of positions to vary with characters from the alphabet
        alphabet (List[str]): List of characters in the alphabet (not including wildcard)
        wildcard_char (str): Character to use as the wildcard (default: '*')
        
    Returns:
        List[str]: List of all features of the given order
    
    Examples:
        >>> get_features_of_order(3, 1, ['A', 'B'])
        ['A**', 'B**', '*A*', '*B*', '**A', '**B']
    """
    # Validate inputs
    if order < 0:
        raise ValueError("Order must be non-negative")
    
    if order > seq_length:
        raise ValueError(f"Order ({order}) cannot exceed sequence length ({seq_length})")
    
    if not alphabet:
        raise ValueError("Alphabet must not be empty")
    
    # Get all combinations of 'order' positions
    orbits = _get_orbits_of_order(seq_length, order)
    
    # Generate features for each orbit and concatenate them using chain.from_iterable
    all_features = list(
        chain.from_iterable(
            _get_features_in_orbit(
                alphabet, 
                positions, 
                seq_length, 
                wildcard_char
            ) 
            for positions in orbits
        )
    )
    
    return all_features 