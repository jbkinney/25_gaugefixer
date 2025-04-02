import numpy as np
import pandas as pd
from typeguard import typechecked

@typechecked
def _additive_series2dict(theta_series:pd.Series, alphabet:list[str], L:int) -> dict:
    features = theta_series.index
    alpha = len(alphabet)
    
    assert alphabet == sorted(alphabet), f'{alphabet=} must be sorted'
    c0 = alphabet[0]

    # Goal: transform features to theta_0, theta_lc. 
    theta_0 = np.float64(0)
    theta_lc = np.zeros((L,alpha), dtype=np.float64)

    # Map features to indices
    features_to_ix_dict = {feature: ix for ix, feature in enumerate(features)}

    # Set 0th order parameter
    feature = ((), '')  
    assert feature in features
    ix = features_to_ix_dict[feature]
    theta_0 = theta_series.values[ix]

    # Set 1st order parameters
    for i in range(L):
        feature = ((i,), c0)
        assert feature in features
        ix = features_to_ix_dict[feature]
        theta_lc[i,:] = theta_series.values[ix:ix+alpha]
                
    # Store as dict
    return dict(theta_0=theta_0, theta_lc=theta_lc, L=L, alpha=alpha, alphabet=alphabet)
