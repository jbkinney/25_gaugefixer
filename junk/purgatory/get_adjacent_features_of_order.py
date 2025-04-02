import numpy as np
from typing import List, Tuple
from itertools import chain
from src.features.get_features_of_order import get_features_of_order
def get_adjacent_features_of_order(
    seq_length: int,
    order: int,
    alphabet: List[str],
    wildcard_char: str = '*'
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
        >>> get_adjacent_features_of_order(seq_length=3, order=2, alphabet=['A', 'B'])
        ['AA*', 'AB*', 'BA*', 'BB*', '*AA', '*AB', '*BA', '*BB']
    """
    
    if order > seq_length:
        raise ValueError(f"Order ({order}) cannot exceed sequence length ({seq_length})")
    
    # Validate inputs
    if order == 0:
        return [wildcard_char*seq_length]
    
    subfeatures = get_features_of_order(L=order, order=order, alphabet=alphabet)
    features = []
    for i in range(seq_length-order+1):
        for subfeature in subfeatures:
            features.append('*'*i + subfeature + '*'*(seq_length-i-order))
    return features
    