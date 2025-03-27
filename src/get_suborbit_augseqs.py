from typing import List
from itertools import chain

from src.get_augseqs_in_orbit import get_augseqs_in_orbit
from src.get_suborbits import get_suborbits

def get_suborbit_augseqs(
    augseq: str,
    alphabet: List[str],
    wildcard_char: str = '*'
) -> List[str]:
    """
    Given an augmented sequence, returns a list of augmented sequences in the corresponding suborbits.
    
    This function takes an augmented sequence and generates all possible augmented sequences
    that can be formed by varying subsets of the non-wildcard positions in the input sequence.
    
    Args:
        augseq (str): The input augmented sequence
        alphabet (List[str]): List of characters in the alphabet (not including wildcard)
        wildcard_char (str): Character used as wildcard (default: '*')
        
    Returns:
        List[str]: List of augmented sequences in the suborbits, sorted by length and lexicographically
        
    Examples:
        >>> get_suborbit_augseqs('*A**', ['A', 'C', 'G', 'T'])
        ['****', '*A**', '*C**', '*G**', '*T**']
    """
    # Get positions of non-wildcard characters
    orbit = [p for p, c in enumerate(augseq) if c != wildcard_char]
    L = len(augseq)
    
    # Get all possible subsets of positions
    suborbits = get_suborbits([orbit])
    
    # Generate augmented sequences for each suborbit
    sp_s = list(chain.from_iterable(
        get_augseqs_in_orbit(alphabet, orbit, L, wildcard_char=wildcard_char)
        for orbit in suborbits
    ))
    
    # Sort by length and lexicographically
    sp_s.sort(key=lambda x: (len(x), x))
    
    return sp_s 