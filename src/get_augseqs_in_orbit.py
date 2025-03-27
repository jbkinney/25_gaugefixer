import numpy as np
import pandas as pd
from typing import List
from itertools import product

def get_augseqs_in_orbit(
    alphabet: List[str],
    positions: List[int],
    seq_length: int,
    wildcard_char: str = '*'
) -> List[str]:
    """
    Generate all augmented sequences of a given order.
    
    This function creates all augmented sequences that have combinations of characters
    from the alphabet at the specified positions, and wildcards at all other positions.
    These sequences form an orbit under the gauge transformation.
    
    Args:
        alphabet (List[str]): List of characters in the alphabet (not including wildcard)
        positions (List[int]): List of positions to vary (0-indexed)
        seq_length (int): Total length of the sequences
        wildcard_char (str): Character to use as the wildcard (default: '*')
        
    Returns:
        List[str]: List of all augmented sequences in the orbit
    
    Examples:
        >>> get_augseqs_in_orbit(['A', 'B', 'C'], [0, 2], 3)
        ['A*A', 'B*A', 'C*A', 'A*B', 'B*B', 'C*B', 'A*C', 'B*C', 'C*C']
    """
    # Validate inputs
    if not set(positions).issubset(set(range(seq_length))):
        raise ValueError(f"Positions must be within range [0, {seq_length-1}]")
    
    if not alphabet:
        raise ValueError("Alphabet must not be empty")
    
    if wildcard_char in alphabet:
        raise ValueError(f"Alphabet should not include the wildcard character '{wildcard_char}'")
    
    # Create a list of character options for each position in the sequence
    char_options = [alphabet if i in positions else [wildcard_char] for i in range(seq_length)]
    
    # Generate all combinations using product and join them into sequences
    aug_sequences = [''.join(chars) for chars in product(*char_options)]
    
    return aug_sequences
