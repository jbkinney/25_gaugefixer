import re
import numpy as np

class SeqEmbedder:
    """
    A class for embedding sequences using features with wildcards.
    
    This class creates regular expression patterns from features with
    wildcards and uses them to embed input sequences into a binary vector space.
    
    Attributes:
        features (list): List of features containing wildcards.
        patterns (list): List of compiled regex patterns derived from features.
    """
    def __init__(self, features, wildcard_char='*'):
        """
        Initialize the SeqEmbedder with features.
        
        Args:
            features (list): List of features containing wildcards.
            wildcard_char (str, optional): Character used as wildcard. Defaults to '*'.
        """
        self.features = features
        self.patterns = [re.compile(sp.replace(wildcard_char,'.')) for sp in features]
        
    def embed(self, seq):
        """
        Embed a sequence into a binary vector space.
        
        Each position in the output array corresponds to whether the input sequence
        matches the corresponding feature pattern.
        
        Args:
            seq (str): The sequence to embed.
            
        Returns:
            numpy.ndarray: Binary vector where each element indicates if the sequence
                           matches the corresponding feature pattern.
        """
        x = [int(p.match(seq) is not None) for p in self.patterns]
        return np.array(x) 