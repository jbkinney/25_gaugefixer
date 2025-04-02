import pandas as pd
import numpy as np
from gaugefixer.features import get_pairwise_features
from typeguard import typechecked

@typechecked
def _pairwise_dict2series(theta_dict: dict) -> pd.Series:
    L = theta_dict['L']
    alphabet = theta_dict['alphabet']
    alpha = len(alphabet)
    theta_0 = theta_dict['theta_0']
    theta_lc = theta_dict['theta_lc']
    theta_lclc = theta_dict['theta_lclc']
    
    assert alphabet == sorted(alphabet), f'{alphabet=} must be sorted'
    c0 = alphabet[0]
    
    # Make features
    features = get_pairwise_features(L=L, alphabet=alphabet)
    
    # Initialize theta_series
    theta_series = pd.Series(index=features, data=0, dtype=np.float64)

    # Map features to indices
    features_to_ix_dict = {feature: ix for ix, feature in enumerate(features)}

    # Set 0th order parameter
    feature = ((), '')  
    assert feature in features
    theta_series[feature] = np.float64(theta_0)

    # Set 1st order parameters
    for i in range(L):
        feature0 = ((i,), c0)
        assert feature0 in features
        ix = features_to_ix_dict[feature0]
        theta_series[ix:ix+alpha] = theta_lc[i,:]

    # Set 2nd order parameters
    for i1 in range(L-1):
        for i2 in range(i1+1,L):
            feature = ((i1,i2), c0+c0)
            assert feature in features
            ix = features_to_ix_dict[feature]
            theta_series[ix:ix+alpha**2] = theta_lclc[i1,:,i2,:].ravel()
            
    return theta_series 