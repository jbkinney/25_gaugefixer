from src.features.petti_feature import PettiFeature
from typeguard import typechecked

@typechecked
def validate_features(features: list[PettiFeature], 
                      L: int, 
                      alphabet: list[str]) -> None:
    """
    Validate the features.
    """
    return True