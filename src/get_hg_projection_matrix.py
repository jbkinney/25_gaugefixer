import numpy as np
import pandas as pd
from scipy import sparse
from typing import List, Union, Tuple

from src.get_suborbit_augseqs import get_suborbit_augseqs

def get_hg_projection_matrix(
    augseqs: List[str],
    alphabet: List[str],
    bg_type: str = 'uniform',
    wt_seq: str | None = None,
    bg_df: pd.DataFrame | None = None,
    wildcard_char: str = '*',
    out_type: str = 'df'
) -> Union[pd.DataFrame, sparse.csr_matrix]:
    """
    Generate the projection matrix for gauge fixing.
    
    This function computes the projection matrix P^λ,p for augmented sequences with '*' wildcards.
    The matrix projects any parameter vector into a specific gauge space.
    
    Args:
        augseqs (List[str]): List of augmented sequences
        alphabet (List[str]): List of characters in the alphabet (not including wildcard)
        bg_type (str): Background model type - either 'uniform' or 'wildtype', or 'custom'
        wt_seq (str): Wildtype sequence; only used if bg_type is 'wildtype'
        bg_df (pd.DataFrame): None or DataFrame of probabilities where:
            - Rows (indices) are sequence positions (0-indexed)
            - Columns are valid characters (excluding wildcard)
            - Values are the probabilities p_l^c
        wildcard_char (str): Character used as wildcard (default: '*')
        out_type (str): Output type - either 'df' or 'sparse' or 'array'
            - 'df': return a pandas DataFrame
            - 'sparse': return a sparse matrix
            - 'array': return a numpy array
        
    Returns:
        Union[pd.DataFrame, sparse.csr_matrix, np.ndarray]: The projection matrix in the requested format
    """
    nonzero_entries = []
    alpha = len(alphabet)
    alpha_inv = 1.0/alpha
    L = len(augseqs[0])
    
    # Check wt_seq if bg_type is 'wildtype'
    if bg_type == 'wildtype':
        if wt_seq is None:
            raise ValueError("wt_seq must be provided if bg_type is 'wildtype'")
        if len(wt_seq) != L:
            raise ValueError("wt_seq must be the same length as the augmented sequences")
    else:
        if wt_seq is not None:
            raise ValueError("wt_seq is not used and must be None if bg_type is not 'wildtype'")
        wt_seq = '*' * L  # just so can be unpacked in the loop

    # Check pi_df if bg_type is 'custom'
    if bg_type == 'custom':
        if bg_df is None:
            raise ValueError("bg_df must be provided if bg_type is 'custom'")
        if bg_df.shape != (L, alpha):
            raise ValueError("bg_df must have the same shape as the augmented sequences")
    else:
        if bg_df is not None:
            raise ValueError("bg_df is not used and must be None if bg_type is not 'custom'")
    
    # Create a lookup dictionary relating augmented sequences to their indices
    augseqs_to_index = {seq: idx for idx, seq in enumerate(augseqs)}
    
    # For each column
    for tp, tp_idx in augseqs_to_index.items():
        # Compute rows with nonzero elements
        sp_s = get_suborbit_augseqs(augseq=tp, alphabet=alphabet, wildcard_char=wildcard_char)
            
        # Compute nonzero elements
        for sp in sp_s:
            sp_idx = augseqs_to_index[sp]
            value = 1.0
            for i, (s, t, wt) in enumerate(zip(sp, tp, wt_seq)):
                if bg_type == 'uniform':
                    pi_t = 1.0 if t == wildcard_char else alpha_inv
                elif bg_type == 'wildtype':
                    pi_t = 1.0 if t == wildcard_char else int(t == wt)
                elif bg_type == 'custom':
                    pi_t = 1.0 if t == wildcard_char else bg_df.at[i, t]
                value *= pi_t if s == wildcard_char else float(s==t) - pi_t
                    
            if value != 0.0:
                nonzero_entries.append((sp_idx, tp_idx, value))
        
    # If not sparse, return DataFrame
    if out_type == 'df':
        df = pd.DataFrame(index=augseqs, columns=augseqs, data=0.0)
        for (sp_idx, tp_idx, value) in nonzero_entries:
            df.iloc[sp_idx, tp_idx] = value
        return df
    # If sparse, return sparse matrix
    elif out_type in ['sparse', 'array']:
        rows, cols, values = zip(*nonzero_entries)
        N = len(augseqs)
        sparse_matrix = sparse.csr_matrix((values, (rows, cols)), shape=(N, N))
        if out_type == 'sparse':
            return sparse_matrix
        elif out_type == 'array':
            return sparse_matrix.todense()
    else:
        raise ValueError(f"Invalid output type: {out_type}. Must be 'df' or 'sparse'") 