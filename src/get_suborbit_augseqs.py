from typing import List
from itertools import chain

from src.get_augseqs_in_orbit import get_augseqs_in_orbit
from src.get_suborbits import get_suborbits
from itertools import product

def get_suborbit_augseqs(
    augseq: str,
    alphabet: List[str],
    wildcard_char: str = '*',
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

    assert len(alphabet)>0, 'Alphabet must not be empty'
    assert not wildcard_char in alphabet, 'Alphabet should not include the wildcard character'
    
    # Get all possible augmented sequences in all the suborbits
    augalphabet = [wildcard_char] + alphabet
    char_lists = [augalphabet if c != wildcard_char else [wildcard_char] for c in augseq]
    sp_s = [''.join(chars) for chars in product(*char_lists)]  
    
    return sp_s 

