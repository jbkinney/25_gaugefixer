from typeguard import typechecked
from src.petti_feature import PettiFeature

@typechecked
def sort_features(features: list[PettiFeature]) -> list[PettiFeature]:
    return sorted(features, key=lambda pf: (len(pf[0]), pf[0], pf[1]))
