from itertools import chain
from src.features.get_features_of_order import get_features_of_order
from src.features.sort_features import sort_features
from src.features.petti_feature import PettiFeature
from typeguard import typechecked

@typechecked
def get_features_upto_order(
    L: int,
    max_order: int,
    alphabet: list[str],
) -> list[PettiFeature]:
    """
    Generate all possible features up to a maximum order for sequences of a specified length.
    
    This function generates features of all orders from 0 up to max_order (inclusive).
    A feature of order k is defined as a specific subsequence (of length k) that appears
    at specific positions in a sequence of length L.
    
    Parameters:
        L (int): The length of the sequences for which features are being generated
        max_order (int): The maximum order of features to generate (0 <= max_order <= L)
        alphabet (list[str]): The set of possible characters that can appear in the sequences
        
    Returns:
        list[PettiFeature]: A sorted list of all features from order 0 to max_order, where each feature is a tuple of:
            - A tuple of positions (integers from 0 to L-1)
            - A string representing the subsequence at those positions
            
    Raises:
        AssertionError: If any of the input validations fail
        
    Example:
        >>> get_features_upto_order(L=2, max_order=1, alphabet=['A', 'B'])
        [((), ''), ((0,), 'A'), ((0,), 'B'), ((1,), 'A'), ((1,), 'B')]
    """

    # Validate inputs
    assert isinstance(L, int), f"seq_length must be an integer, not {type(L)}"
    assert isinstance(max_order, int), f"max_order must be an integer, not {type(max_order)}"
    assert isinstance(alphabet, list), f"alphabet must be a list, not {type(alphabet)}"
    assert len(alphabet) > 0, "alphabet must not be empty"
    assert all(isinstance(a, str) for a in alphabet), "alphabet must contain only strings"
    assert max_order >= 0, "max_order must be non-negative"
    assert max_order <= L, f"max_order ({max_order}) must be less than or equal to seq_length ({L})"

    # Generate features for each order from 0 to max_order and concatenate them
    features = list(
        chain.from_iterable(
            get_features_of_order(L, order, alphabet)
            for order in range(max_order + 1)
        )
    )
    
    # Sort features
    features = sort_features(features)
    
    return features