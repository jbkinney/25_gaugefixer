import numpy as np
import pandas as pd
from typeguard import typechecked

@typechecked
def _pairwise_series2dict(theta_series:pd.Series, alphabet:list[str], L:int) -> dict:
    features = theta_series.index
    alpha = len(alphabet)
    
    assert alphabet == sorted(alphabet), f'{alphabet=} must be sorted'
    c0 = alphabet[0]

    # Goal: transform features to theat_0, theta_lc, theta_lclc. 
    theta_0 = np.float64(0)
    theta_lc = np.zeros((L,alpha), dtype=np.float64)
    theta_lclc = np.zeros((L,alpha,L,alpha),  dtype=np.float64)

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

    # Set 2nd order parameters
    for i1 in range(L-1):
        for i2 in range(i1+1,L):
            feature = ((i1,i2), c0+c0)
            assert feature in features
            ix = features_to_ix_dict[feature]
            theta_lclc[i1,:,i2,:] = theta_series.values[ix:ix+alpha**2].reshape(alpha,alpha)
            
                
    # Store as dict
    return dict(theta_0=theta_0, theta_lc=theta_lc, theta_lclc=theta_lclc, L=L, alpha=alpha, alphabet=alphabet) 