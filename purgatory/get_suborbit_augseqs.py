from purgatory._get_features_in_orbit import _get_features_in_orbit
from purgatory._get_suborbits import _get_suborbits
from itertools import product

def get_suborbit_augseqs(
    augseq: str,
    alphabet: list[str],
    wildcard_char: str = '*',
) -> list[str]:
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

    assert isinstance(augseq, str), 'augseq must be a str'
    assert isinstance(alphabet, list), 'alphabet must be a list'
    assert isinstance(wildcard_char, str), 'wildcard_char must be a str'
    assert len(augseq)>0, 'augseq must have length > 0'
    assert len(alphabet)>0, 'Alphabet must not be empty'
    assert len(wildcard_char)==1, 'wildcard_char must be of length 1'
    assert not wildcard_char in alphabet, 'Alphabet should not include the wildcard character'
    
    # Get all possible augmented sequences in all the suborbits
    augalphabet = (wildcard_char,) + tuple(alphabet)
    wildcard_only = (wildcard_char,)
    char_lists = [augalphabet if c != wildcard_char else wildcard_only for c in augseq]
    sp_s = [''.join(chars) for chars in product(*char_lists)]  
    
    return sp_s 

