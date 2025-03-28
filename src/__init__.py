from src.get_features_upto_order import get_features_upto_order
from src.get_adjacent_features_upto_order import get_adjacent_features_upto_order

import numpy as np
_alphabet_dict = {
    'dna': ['A', 'C', 'G', 'T'],
    'rna': ['A', 'C', 'G', 'U'],
    'protein': ['A', 'C', 'D', 'E', 'F',
                'G', 'H', 'I', 'K', 'L',
                'M', 'N', 'P', 'Q', 'R',
                'S', 'T', 'V', 'W', 'Y'],
    'protein~': ['A', 'C', 'D', 'E', 'F',
                 'G', 'H', 'I', 'K', 'L',
                 'M', 'N', 'P', 'Q', 'R',
                 'S', 'T', 'V', 'W', 'Y', '~']
}

def get_alphabet(alphabet_type: str) -> np.ndarray:
    """Get the alphabet for a given alphabet type.
    
    Args:
        alphabet_type (str): The type of alphabet to get.

    Returns:
        np.ndarray: The alphabet for the given alphabet type.
    """
    if alphabet_type not in _alphabet_dict:
        raise ValueError(f"Invalid alphabet type: {alphabet_type}")
    return _alphabet_dict[alphabet_type]

def get_additive_model_features(
    L: int,
    alphabet: list[str],
    wildcard_char: str = '*'):
    return get_features_upto_order(seq_length=L, 
                                   max_order=1, 
                                   alphabet=alphabet, 
                                   wildcard_char=wildcard_char)
    
def get_pairwise_model_features(
    L: int,
    alphabet: list[str],
    wildcard_char: str = '*'):
    return get_features_upto_order(seq_length=L, 
                                   max_order=2, 
                                   alphabet=alphabet, 
                                   wildcard_char=wildcard_char)

def get_neighbor_model_features(
    L: int,
    alphabet: list[str],
    wildcard_char: str = '*'):
    return get_adjacent_features_upto_order(seq_length=L, 
                                            order=2, 
                                            alphabet=alphabet, 
                                            wildcard_char=wildcard_char)
    
def get_all_order_model_features(
    L: int,
    alphabet: list[str],
    wildcard_char: str = '*'):
    return get_features_upto_order(seq_length=L, 
                                   max_order=L, 
                                   alphabet=alphabet, 
                                   wildcard_char=wildcard_char)

def get_K_order_model_features(
    L: int,
    K: int,
    alphabet: list[str],
    wildcard_char: str = '*'):
    return get_features_upto_order(seq_length=L, 
                                   max_order=K, 
                                   alphabet=alphabet, 
                                   wildcard_char=wildcard_char)

def get_K_adjacent_model_features(
    L: int,
    K: int,
    alphabet: list[str],
    wildcard_char: str = '*'):
    return get_adjacent_features_upto_order(seq_length=L, 
                                            order=K, 
                                            alphabet=alphabet, 
                                            wildcard_char=wildcard_char)