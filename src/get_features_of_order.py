from itertools import product
from src.sort_features import sort_features
from src.petti_feature import PettiFeature
from typeguard import typechecked

@typechecked
def get_features_of_order(
    seq_length: int,
    order: int,
    alphabet: list[str],
) -> list[PettiFeature]:

    # Validate inputs
    assert isinstance(order, int), f"order must be an integer, not {type(order)}"
    assert isinstance(seq_length, int), f"seq_length must be an integer, not {type(seq_length)}"
    assert isinstance(alphabet, list), f"alphabet must be a list, not {type(alphabet)}"
    assert len(alphabet) > 0, "alphabet must not be empty"
    assert all(isinstance(a, str) for a in alphabet), "alphabet must contain only strings"
    assert order >= 0, "order must be non-negative"
    assert seq_length > 0, "seq_length must be positive"
    assert order <= seq_length, f"order ({order}) must be less than or equal to seq_length ({seq_length})"
    
    # Get all possible combinations of order
    from itertools import combinations
    position_combinations = list(combinations(range(seq_length), order))
    
    # Get all possible subsequences of order
    possible_subsequences = [''.join(p) for p in product(alphabet, repeat=order)]
    
    # Create all combinations of positions and subsequences
    features = list(product(position_combinations, possible_subsequences))
        
    # Sort features
    features = sort_features(features)
    
    return features 