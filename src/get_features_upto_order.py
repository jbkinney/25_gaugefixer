from itertools import chain
from src.get_features_of_order import get_features_of_order
from src.sort_features import sort_features
from src.petti_feature import PettiFeature
from typeguard import typechecked

@typechecked
def get_features_upto_order(
    seq_length: int,
    max_order: int,
    alphabet: list[str],
) -> list[PettiFeature]:

    # Validate inputs
    assert isinstance(seq_length, int), f"seq_length must be an integer, not {type(seq_length)}"
    assert isinstance(max_order, int), f"max_order must be an integer, not {type(max_order)}"
    assert isinstance(alphabet, list), f"alphabet must be a list, not {type(alphabet)}"
    assert len(alphabet) > 0, "alphabet must not be empty"
    assert all(isinstance(a, str) for a in alphabet), "alphabet must contain only strings"
    assert max_order >= 0, "max_order must be non-negative"
    assert max_order <= seq_length, f"max_order ({max_order}) must be less than or equal to seq_length ({seq_length})"

    # Generate features for each order from 0 to max_order and concatenate them
    features = list(
        chain.from_iterable(
            get_features_of_order(seq_length, order, alphabet)
            for order in range(max_order + 1)
        )
    )
    
    # Sort features
    features = sort_features(features)
    
    return features