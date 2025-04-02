import numpy as np
from typing import List, Tuple
from src.features.get_features_of_order import get_features_of_order
from src.features.sort_features import sort_features

def get_adjacent_features_upto_order(
    seq_length: int,
    order: int,
    alphabet: List[str],
    wildcard_char: str = '*',
) -> List[Tuple[int, ...]]:
    """
    """
    
    if order > seq_length:
        raise ValueError(f"Order ({order}) cannot exceed sequence length ({seq_length})")
    
    # If order = 0, just return stars
    if order == 0:
        features = [wildcard_char*seq_length]
        
    # If order = 1, return additive features
    if order >= 1:
        augalphabet = ['*'] + alphabet
        subfeatures = get_features_of_order(L=order, 
                                            order=order, 
                                            alphabet=augalphabet)
        
        features = []
        for i in range(seq_length-order+1):
            for subfeature in subfeatures:
                features.append('*'*i + subfeature + '*'*(seq_length-i-order))
                
        # Remove features
        features = list(set(features))

        # Sort features
        features = sort_features(features)
            
    return features