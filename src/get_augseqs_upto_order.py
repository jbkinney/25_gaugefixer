import numpy as np
import pandas as pd
from typing import List
from itertools import chain

from src.get_augseqs_of_order import get_augseqs_of_order

def get_augseqs_upto_order(
    seq_length: int,
    max_order: int,
    alphabet: List[str],
    wildcard_char: str = '*'
) -> List[str]:
    """
    Generate all augmented sequences of orders from 0 up to and including max_order.
    
    This function generates all augmented sequences where at most 'max_order' positions
    have characters from the alphabet, and all other positions have wildcards.
    
    Args:
        seq_length (int): The total length of the sequence
        max_order (int): The maximum number of positions to vary with characters
        alphabet (List[str]): List of characters in the alphabet (not including wildcard)
        wildcard_char (str): Character to use as the wildcard (default: '*')
        
    Returns:
        List[str]: List of all augmented sequences of orders 0 to max_order
    
    Examples:
        >>> get_augseqs_upto_order(2, 1, ['A', 'B'])
        ['**', 'A*', 'B*', '*A', '*B']
    """
    # Validate inputs
    if max_order < 0:
        raise ValueError("Maximum order must be non-negative")
    
    if max_order > seq_length:
        raise ValueError(f"Maximum order ({max_order}) cannot exceed sequence length ({seq_length})")
    
    # Generate augmented sequences for each order from 0 to max_order and concatenate them
    all_augseqs = list(
        chain.from_iterable(
            get_augseqs_of_order(seq_length, order, alphabet, wildcard_char)
            for order in range(max_order + 1)
        )
    )
    
    return all_augseqs 