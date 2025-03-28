import re
import numpy as np

class SeqEmbedder:
    """
    A class for embedding sequences using augmented sequences with wildcards.
    
    This class creates regular expression patterns from augmented sequences with
    wildcards and uses them to embed input sequences into a binary vector space.
    
    Attributes:
        augseqs (list): List of augmented sequences containing wildcards.
        patterns (list): List of compiled regex patterns derived from augseqs.
    """
    def __init__(self, augseqs, wildcard_char='*'):
        """
        Initialize the SeqEmbedder with augmented sequences.
        
        Args:
            augseqs (list): List of augmented sequences containing wildcards.
            wildcard_char (str, optional): Character used as wildcard. Defaults to '*'.
        """
        self.augseqs = augseqs
        self.patterns = [re.compile(sp.replace(wildcard_char,'.')) for sp in augseqs]
        
    def embed(self, seq):
        """
        Embed a sequence into a binary vector space.
        
        Each position in the output array corresponds to whether the input sequence
        matches the corresponding augmented sequence pattern.
        
        Args:
            seq (str): The sequence to embed.
            
        Returns:
            numpy.ndarray: Binary vector where each element indicates if the sequence
                           matches the corresponding augmented sequence pattern.
        """
        x = [int(p.match(seq) is not None) for p in self.patterns]
        return np.array(x) 