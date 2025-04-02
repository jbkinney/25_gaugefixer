import numpy as np
import pandas as pd
from scipy import sparse
from typing import List, Union, Tuple
from src.features.petti_feature import PettiFeature
from src.features._get_suborbit_features import _get_suborbit_features
from typeguard import typechecked

@typechecked
def fix_hg_series(
    series: pd.Series,
    L: int,
    alphabet: list[str],
    bg_df: pd.DataFrame,
) -> pd.Series:

    features = series.index
    fixed_series = pd.Series(index=features, data=0.0)
    
    # For each column
    for tp_feature, theta_tp in series.items():
        # Get the orbit and subsequence of tp
        tp_orbit, tp_subseq = tp_feature
        
        # Compute rows with nonzero elements
        sp_features = _get_suborbit_features(feature=tp_feature, alphabet=alphabet)
            
        # Compute nonzero elements
        for sp_feature in sp_features:
            
            # Initialize value
            coef = 1.0
            
            # Get the orbit and subsequence of sp
            sp_orbit, sp_subseq = sp_feature
            
            # Compute the value of the projection matrix element
            # TODO: This requires reworking. 
            for i in range(L):
                tp_char = tp_subseq[tp_orbit.index(i)] if i in tp_orbit else 'wildcard'
                sp_char = sp_subseq[sp_orbit.index(i)] if i in sp_orbit else 'wildcard'
                pi_t = 1.0 if tp_char == 'wildcard' else bg_df.at[i, tp_char]
                coef *= pi_t if i not in sp_orbit else float(sp_char==tp_char) - pi_t
                    
            if coef != 0.0:
                fixed_series.at[sp_feature] += coef * theta_tp
        
    return fixed_series