from itertools import product
from src.features.petti_feature import PettiFeature
from typeguard import typechecked

@typechecked
def _get_suborbit_features(
    feature: PettiFeature,
    alphabet: list[str],
) -> list[PettiFeature]:
    """
    Given a feature, returns a list of features in the corresponding suborbits.
    
    This function takes a feature and generates all possible features
    that can be formed by varying subsets of the non-wildcard positions in the input sequence.
    
    Args:
        feature (str): The input feature
        alphabet (List[str]): List of characters in the alphabet (not including wildcard)
        wildcard_char (str): Character used as wildcard (default: '*')
        
    Returns:
        List[str]: List of features in the suborbits, sorted by length and lexicographically
        
    Examples:
        >>> get_suborbit_features('*A**', ['A', 'C', 'G', 'T'])
        ['****', '*A**', '*C**', '*G**', '*T**']
    """
    
    # Get all possible features in all the suborbits
    orbits, subsequence = feature
    
    # Get all possible subsets of the orbit positions
    suborbits = []
    for r in range(len(orbits) + 1):
        from itertools import combinations
        suborbits.extend(combinations(orbits, r))
        
    # Initialize list to store all subfeatures
    out_features = []
    
    # For each suborbit, generate all possible subsequences of that length
    for suborbit in suborbits:
        # Get length of current suborbit
        suborbit_len = len(suborbit)
        
        # Generate all possible subsequences of that length using alphabet
        possible_subsequences = [''.join(p) for p in product(alphabet, repeat=suborbit_len)]
        
        # Create features by pairing suborbit with each subsequence
        suborbit_features = [(suborbit, subseq) for subseq in possible_subsequences]
        
        # Add these features to our list
        out_features.extend(suborbit_features)
    
    return out_features 

