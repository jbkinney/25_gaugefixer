from itertools import product
from src.sort_features import sort_features
from src.petti_feature import PettiFeature
from typeguard import typechecked

@typechecked
def get_features_of_order(
    L: int,
    order: int,
    alphabet: list[str],
) -> list[PettiFeature]:
    """
    Generate all possible features of a given order for sequences of a specified length.
    
    A feature of order k is defined as a specific subsequence (of length k) that appears
    at specific positions in a sequence of length L.
    
    Parameters:
        L (int): The length of the sequences for which features are being generated
        order (int): The order/length of the subsequences to consider (0 <= order <= L)
        alphabet (list[str]): The set of possible characters that can appear in the sequences
        
    Returns:
        list[PettiFeature]: A sorted list of features, where each feature is a tuple of:
            - A tuple of positions (integers from 0 to L-1)
            - A string representing the subsequence at those positions
            
    Raises:
        AssertionError: If any of the input validations fail
    """

    # Validate inputs
    assert isinstance(order, int), f"order must be an integer, not {type(order)}"
    assert isinstance(L, int), f"seq_length must be an integer, not {type(L)}"
    assert isinstance(alphabet, list), f"alphabet must be a list, not {type(alphabet)}"
    assert len(alphabet) > 0, "alphabet must not be empty"
    assert all(isinstance(a, str) for a in alphabet), "alphabet must contain only strings"
    assert order >= 0, "order must be non-negative"
    assert L > 0, "seq_length must be positive"
    assert order <= L, f"order ({order}) must be less than or equal to seq_length ({L})"
    
    # Get all possible combinations of order
    from itertools import combinations
    position_combinations = list(combinations(range(L), order))
    
    # Get all possible subsequences of order
    possible_subsequences = [''.join(p) for p in product(alphabet, repeat=order)]
    
    # Create all combinations of positions and subsequences
    features = list(product(position_combinations, possible_subsequences))
        
    # Sort features
    features = sort_features(features)
    
    return features 