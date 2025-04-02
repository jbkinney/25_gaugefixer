import re
import numpy as np
from typeguard import typechecked
from gaugefixer.features.petti_feature import PettiFeature

class SeqEmbedder:
    """
    A class for embedding sequences into a feature space defined by pattern matching.
    
    This class creates binary feature vectors for sequences based on matches to patterns
    derived from specified features. Each feature consists of a tuple of positions (orbit)
    and a subsequence that should appear at those positions.
    
    Attributes:
        L (int): The length of sequences that can be embedded.
        features (list): List of features, where each feature is a tuple of 
                         (position_tuple, subsequence).
        patterns (list): List of compiled regular expressions derived from the features.
    """
    
    @typechecked
    def __init__(self, features: list[PettiFeature], L: int, check_features: bool = True):
        """
        Initialize the SeqEmbedder with features and sequence length.
        
        Args:
            features (list): List of features, where each feature is a tuple of 
                            (position_tuple, subsequence). The position_tuple indicates 
                            the positions in the sequence where the subsequence should match.
            L (int): The length of sequences that will be embedded.
            check_features (bool, optional): Whether to validate the features. Defaults to True.
                                           If False, skips validation for better performance.
            
        Raises:
            AssertionError: If any of the input validations fail.
            TypeError: If inputs are not of the expected types.
        """
        # Validate inputs
        assert isinstance(L, int), f"L must be an integer, not {type(L)}"
        assert L > 0, "L must be positive"
        assert isinstance(features, list), f"features must be a list, not {type(features)}"
        assert all(isinstance(f, tuple) and len(f) == 2 for f in features), \
            "Each feature must be a tuple of (position_tuple, subsequence)"
        
        self.L = L
        self.features = features
        
        if not check_features:
            self.patterns = [re.compile(''.join([subseq[orbit.index(i)] if i in orbit else '.' for i in range(L)])) for orbit, subseq in features]
        else:
            # Create regex patterns from features
            self.patterns = []
            for orbit, subseq in features:
                # Validate each feature
                if orbit:  # Skip validation for order-0 features with empty orbit
                    assert all(isinstance(i, int) and 0 <= i < L for i in orbit), \
                        f"Position indices in orbit must be integers in range [0, {L-1}]"
                    assert len(subseq) == len(orbit), \
                        f"Subsequence length ({len(subseq)}) must match orbit length ({len(orbit)})"
                
                # Create the pattern
                pattern = ['.' for _ in range(L)]
                for i, pos in enumerate(orbit):
                    pattern[pos] = subseq[i]
                self.patterns.append(re.compile(''.join(pattern)))
        
    def embed(self, seq: str) -> np.ndarray:
        """
        Embed a sequence into the feature space.
        
        For each pattern derived from the features, this method checks if the 
        pattern matches the input sequence. The result is a binary vector where
        each element indicates whether the corresponding pattern matched.
        
        Args:
            seq (str): The input sequence to embed. Must be of length L.
            
        Returns:
            numpy.ndarray: A binary vector where each element is 1 if the corresponding
                          pattern matched the sequence, 0 otherwise.
                          
        Example:
            >>> embedder = SeqEmbedder(features=[((0,), 'A')], L=3)
            >>> embedder.embed('ABC')
            array([1])
            >>> embedder.embed('TGC')
            array([0])
            
        Raises:
            ValueError: If the sequence length doesn't match the expected length L.
        """
        # Validate input sequence
        if not isinstance(seq, str):
            raise TypeError(f"Sequence must be a string, not {type(seq)}")
        
        if len(seq) != self.L:
            raise ValueError(f"Sequence length ({len(seq)}) must match the expected length ({self.L})")
        
        # Perform embedding
        x = [int(p.match(seq) is not None) for p in self.patterns]
        return np.array(x) 