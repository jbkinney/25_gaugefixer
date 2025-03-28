import numpy as np
import pandas as pd
from typing import Union, Optional, List

def gauge_fix_sequences(
    df: pd.DataFrame,
    lambda_param: Union[float, str],
    p_lc: pd.DataFrame,
    sequence_col: str = 'sequence',
    theta_col: str = 'theta',
    wildcard: str = '*'
) -> pd.DataFrame:
    """
    Apply gauge fixing to sequences and their corresponding parameters using Eq. 19 (eq:projection_matrix)
    from the paper. Implements the projection matrix P^λ,p for augmented sequences with '*' wildcards.
    
    Args:
        df (pd.DataFrame): Input dataframe containing augmented sequences and parameters
        lambda_param (Union[float, str]): Lambda parameter for gauge fixing. Can be a float or 'inf'
        p_lc (pd.DataFrame): DataFrame of p_l^c probabilities where:
                           - Rows (indices) are sequence positions l (0-indexed)
                           - Columns are valid characters c (excluding the wildcard character)
                           - Values are the probabilities p_l^c
        sequence_col (str): Name of the column containing sequences
        theta_col (str): Name of the column containing parameters theta
        wildcard (str): Character used as wildcard (default: '*')
        
    Returns:
        pd.DataFrame: DataFrame with gauge-fixed sequences and parameters
    """
    # Convert lambda to float if it's 'inf'
    if lambda_param == 'inf':
        lambda_param = float('inf')
    
    # Calculate eta = λ/(1+λ)
    if lambda_param == float('inf'):
        eta = 1.0
    else:
        eta = lambda_param / (1.0 + lambda_param)
    
    # Get unique sequences and their indices
    unique_sequences = df[sequence_col].unique()
    n_sequences = len(unique_sequences)
    
    # Get the sequence length from the first sequence
    first_seq = unique_sequences[0]
    L = len(first_seq)
    
    # Make sure all sequences have the same length
    for seq in unique_sequences:
        if len(seq) != L:
            raise ValueError(f"All sequences must have the same length. Found {len(seq)} != {L}")
    
    # Check that sequence length matches probability matrix positions
    if L != len(p_lc):
        raise ValueError(f"Sequence length {L} does not match probability matrix rows {len(p_lc)}")
    
    # Create mapping from sequence to index for easy lookup
    seq_to_idx = {seq: idx for idx, seq in enumerate(unique_sequences)}
    
    # Initialize the projection matrix P^λ,p
    P_matrix = np.zeros((n_sequences, n_sequences))
    
    # Compute the projection matrix according to eq:projection_matrix
    for i, s_prime in enumerate(unique_sequences):
        for j, t_prime in enumerate(unique_sequences):
            # Initialize the matrix element to 1 (for product calculation)
            P_element = 1.0
            
            for l in range(L):
                s_l = s_prime[l]
                t_l = t_prime[l]
                
                # Get probability of character t_l at position l (if it's not a wildcard)
                p_l_t = p_lc.loc[l, t_l] if t_l != wildcard and t_l in p_lc.columns else 0.0
                
                # Case 1: s'_l ∈ A and t'_l ∈ A (both are characters, not wildcards)
                if s_l != wildcard and t_l != wildcard:
                    if s_l == t_l:
                        P_element *= (1 - p_l_t * eta)
                    else:
                        P_element *= (-p_l_t * eta)
                
                # Case 2: s'_l = * and t'_l ∈ A (s has wildcard, t has character)
                elif s_l == wildcard and t_l != wildcard:
                    P_element *= (p_l_t * eta)
                
                # Case 3: s'_l ∈ A and t'_l = * (s has character, t has wildcard)
                elif s_l != wildcard and t_l == wildcard:
                    P_element *= (1 - eta)
                
                # Case 4: s'_l = * and t'_l = * (both are wildcards)
                elif s_l == wildcard and t_l == wildcard:
                    P_element *= eta
            
            P_matrix[i, j] = P_element
    
    # Get the original theta values
    theta_values = df.groupby(sequence_col)[theta_col].first().values
    
    # Apply gauge fixing transformation: θ_fixed = P^λ,p @ θ
    fixed_theta = P_matrix @ theta_values
    
    # Create new dataframe with fixed values
    result_df = df.copy()
    for seq, fixed_val in zip(unique_sequences, fixed_theta):
        mask = result_df[sequence_col] == seq
        result_df.loc[mask, theta_col] = fixed_val
    
    return result_df 