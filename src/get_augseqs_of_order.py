import numpy as np
import pandas as pd
from typing import List
from itertools import chain

from src.get_orbits_of_order import get_orbits_of_order
from src.get_augseqs_in_orbit import get_augseqs_in_orbit

def get_augseqs_of_order(
    seq_length: int,
    order: int,
    alphabet: List[str],
    wildcard_char: str = '*'
) -> List[str]:
    """
    Generate all augmented sequences of a given order.
    
    This function generates all augmented sequences where exactly 'order' positions
    have characters from the alphabet, and all other positions have wildcards.
    It first finds all possible combinations of 'order' positions, then generates
    all possible augmented sequences for each combination.
    
    Args:
        seq_length (int): The total length of the sequence
        order (int): The number of positions to vary with characters from the alphabet
        alphabet (List[str]): List of characters in the alphabet (not including wildcard)
        wildcard_char (str): Character to use as the wildcard (default: '*')
        
    Returns:
        List[str]: List of all augmented sequences of the given order
    
    Examples:
        >>> get_augseqs_of_order(3, 1, ['A', 'B'])
        ['A**', 'B**', '*A*', '*B*', '**A', '**B']
    """
    # Validate inputs
    if order < 0:
        raise ValueError("Order must be non-negative")
    
    if order > seq_length:
        raise ValueError(f"Order ({order}) cannot exceed sequence length ({seq_length})")
    
    if not alphabet:
        raise ValueError("Alphabet must not be empty")
    
    if wildcard_char in alphabet:
        raise ValueError(f"Alphabet should not include the wildcard character '{wildcard_char}'")
    
    # Get all combinations of 'order' positions
    orbits = get_orbits_of_order(seq_length, order)
    
    # Generate augmented sequences for each orbit and concatenate them using chain.from_iterable
    all_augseqs = list(
        chain.from_iterable(
            get_augseqs_in_orbit(
                alphabet, 
                positions, 
                seq_length, 
                wildcard_char
            ) 
            for positions in orbits
        )
    )
    
    return all_augseqs 