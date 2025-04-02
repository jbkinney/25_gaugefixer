# Main package init file 
from gaugefixer.features.get_features_upto_order import get_features_upto_order

import numpy as np
_alphabet_dict = {
    'dna': ['A', 'C', 'G', 'T'],
    'rna': ['A', 'C', 'G', 'U'],
    'protein': ['A', 'C', 'D', 'E', 'F',
                'G', 'H', 'I', 'K', 'L',
                'M', 'N', 'P', 'Q', 'R',
                'S', 'T', 'V', 'W', 'Y'],
    'protein*': ['A', 'C', 'D', 'E', 'F',
                 'G', 'H', 'I', 'K', 'L',
                 'M', 'N', 'P', 'Q', 'R',
                 'S', 'T', 'V', 'W', 'Y', '*']
}

def get_alphabet(alphabet_type: str) -> list[str]:
    """Get the alphabet for a given alphabet type.
    
    Args:
        alphabet_type (str): The type of alphabet to get.

    Returns:
        list[str]: The alphabet for the given alphabet type.
    """
    if alphabet_type not in _alphabet_dict:
        raise ValueError(f"Invalid alphabet type: {alphabet_type}")
    return _alphabet_dict[alphabet_type] 