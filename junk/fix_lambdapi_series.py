import pandas as pd
import numpy as np
from typing import Union
from scipy import sparse
from src import get_allorder_features
import itertools
from src.features.sort_features import sort_features
from src.fixers._kron_matvec import _kron_matvec

def fix_lambdapi_series(
    series: pd.Series,
    lam: float,
    pi_lc: np.array,  
    L: int, 
    alphabet: list[str],
    ) -> pd.Series:

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
        
    # Compute corresponding augmented sequence features
    augalphabet = ['*']+alphabet
    augseqs = [''.join(seq) for seq in itertools.product(augalphabet, repeat=L)]
        
    # Convert augmented sequences to PettiFeatures
    features_in_kron_order = []
    for augseq in augseqs:
        # Create orbit tuple from positions of non-* characters
        orbit = tuple(i for i, c in enumerate(augseq) if c != '*')
        # Create subsequence string from non-* characters
        subseq = ''.join(c for c in augseq if c != '*')
        features_in_kron_order.append((orbit, subseq))
        
    in_features = list(series.index)
        
    series = series.copy()
    series = series.reindex(features_in_kron_order)
    fixed_values = _kron_matvec(factor_matrices, series.values)
    fixed_series = pd.Series(data=fixed_values, index=features_in_kron_order)
    fixed_series = fixed_series.reindex(in_features)
    
    return fixed_series