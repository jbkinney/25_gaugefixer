from typeguard import typechecked
from src.features.petti_feature import PettiFeature

@typechecked
def sort_features(features: list[PettiFeature]) -> list[PettiFeature]:
    """
    Sort a list of PettiFeature objects in a consistent order.
    
    The sorting is performed with the following priority:
    1. By length of the first tuple (orbit size)
    2. By the values in the first tuple (orbit elements)
    3. By the string value (label)
    
    Parameters
    ----------
    features : list[PettiFeature]
        A list of PettiFeature objects, each a tuple of (tuple[int, ...], str)
        
    Returns
    -------
    list[PettiFeature]
        The sorted list of PettiFeature objects
    
    Examples
    --------
    >>> sort_features([((1, 2), "ab"), ((1,), "c"), ((1, 2), "aa")])
    [((1,), "c"), ((1, 2), "aa"), ((1, 2), "ab")]
    """
    return sorted(features, key=lambda pf: (len(pf[0]), pf[0], pf[1]))
