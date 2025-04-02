import numpy as np
import pandas as pd
from scipy import sparse
from typing import List, Union, Tuple
from src.features.petti_feature import PettiFeature
from src.features._get_suborbit_features import _get_suborbit_features
from typeguard import typechecked

@typechecked
def get_hg_matrix(
    features: list[PettiFeature],
    L: int,
    alphabet: list[str],
    bg_df: pd.DataFrame,
    out_type: str = 'df'
) -> Union[pd.DataFrame, sparse.csr_matrix, np.ndarray]:

    nonzero_entries = []
    alpha = len(alphabet)
    
    # Create a lookup dictionary relating features to their indices
    features_to_index = {feature: idx for idx, feature in enumerate(features)}
    
    # For each column
    for tp_feature, tp_idx in features_to_index.items():
        # Get the orbit and subsequence of tp
        tp_orbit, tp_subseq = tp_feature
        
        # Compute rows with nonzero elements
        sp_features = _get_suborbit_features(feature=tp_feature, alphabet=alphabet)
            
        # Compute nonzero elements
        for sp_feature in sp_features:
            sp_idx = features_to_index[sp_feature]
            value = 1.0
            
            # Get the orbit and subsequence of sp
            sp_orbit, sp_subseq = sp_feature
            
            # Compute the value of the projection matrix element
            # TODO: This requires reworking. 
            for i in range(L):
                
                tp_char = tp_subseq[tp_orbit.index(i)] if i in tp_orbit else '*'
                sp_char = sp_subseq[sp_orbit.index(i)] if i in sp_orbit else '*'
                pi_t = 1.0 if tp_char == '*' else bg_df.at[i, tp_char]
                value *= pi_t if i not in sp_orbit else float(sp_char==tp_char) - pi_t
                    
            if value != 0.0:
                nonzero_entries.append((sp_idx, tp_idx, value))
        
    # If not sparse, return DataFrame
    if out_type == 'df':
        df = pd.DataFrame(index=features, columns=features, data=0.0)
        for (sp_idx, tp_idx, value) in nonzero_entries:
            df.iloc[sp_idx, tp_idx] = value
        return df
    # If sparse, return sparse matrix
    elif out_type in ['sparse', 'array']:
        rows, cols, values = zip(*nonzero_entries)
        N = len(features)
        sparse_matrix = sparse.csr_matrix((values, (rows, cols)), shape=(N, N))
        if out_type == 'sparse':
            return sparse_matrix
        elif out_type == 'array':
            return sparse_matrix.toarray()
    else:
        raise ValueError(f"Invalid output type: {out_type}. Must be 'df' or 'sparse'") 