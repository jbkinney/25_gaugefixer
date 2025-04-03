import pandas as pd
import numpy as np
from typing import Union
from scipy import sparse
import itertools
from gaugefixer.features.sort_features import sort_features

def get_allorder_matrix(
    lam: float,
    pi_lc: np.array,  
    L: int, 
    alphabet: list[str],
    out_type: str = 'df'
    ) -> Union[pd.DataFrame, np.ndarray]:

    """"
    Fix a series of parameters in the all-order interaction model.
    """
    
    assert lam >= 0, "Lambda must be non-negative"
    if np.isfinite(lam):
        eta = lam/(1+lam)
    elif lam == np.inf:
        eta = 1.0
    else:
        raise ValueError(f"Invalid lambda value: {lam}")
    
    alpha = len(alphabet)
    
    factor_matrices = []
    for i in range(L):
        mat = np.zeros((alpha+1,alpha+1))
        mat[0,0] = eta
        mat[1:,0] = 1-eta
        mat[0,1:] = eta*pi_lc[i,:]
        mat[1:,1:] = np.eye(alpha) - eta*pi_lc[i,:][np.newaxis,:]
        factor_matrices.append(mat)
        
    # Start with first matrix and iteratively compute Kronecker product
    result = factor_matrices[0]
    for i in range(1, len(factor_matrices)):
        result = np.kron(result, factor_matrices[i])
        
    # Compute corresponding augmented sequence features
    augalphabet = ['*']+alphabet
    augseqs = [''.join(seq) for seq in itertools.product(augalphabet, repeat=L)]
        
    # Convert augmented sequences to PettiFeatures
    features = []
    for augseq in augseqs:
        # Create orbit tuple from positions of non-* characters
        orbit = tuple(i for i, c in enumerate(augseq) if c != '*')
        # Create subsequence string from non-* characters
        subseq = ''.join(c for c in augseq if c != '*')
        features.append((orbit, subseq))
        
    out_df = pd.DataFrame(index=features, columns=features, data=result)
    
    # Sort rows and columns of out_df using sort_features
    sorted_features = sort_features(features)
    out_df = out_df.loc[sorted_features, sorted_features]
    
    if out_type == 'df':
        return out_df
    elif out_type == 'array':
        return out_df.to_numpy()
    else:
        raise ValueError(f"Invalid output type: {out_type}") 