import pandas as pd
import numpy as np
from src import get_pairwise_model_features

def pairwise_theta_dict_to_series(theta_dict):
    L = theta_dict['L']
    alphabet = theta_dict['alphabet']
    alpha = len(alphabet)
    theta_0 = theta_dict['theta_0']
    theta_lc = theta_dict['theta_lc']
    theta_lclc = theta_dict['theta_lclc']
    
    assert alphabet == sorted(alphabet), f'{alphabet=} must be sorted'
    c0 = alphabet[0]
    
    # Make features
    features = get_pairwise_model_features(L=L, alphabet=alphabet)
    
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


# def pairwise_theta_dict_to_series(theta_dict):
#     L = theta_dict['L']
#     alphabet = theta_dict['alphabet']
#     theta_0 = theta_dict['theta_0']
#     theta_lc = theta_dict['theta_lc']
#     theta_lclc = theta_dict['theta_lclc']
    
#     # Make features
#     features = get_pairwise_model_features(L=L, alphabet=alphabet)
    
#     # Initialize theta_series
#     theta_series = pd.Series(index=features, data=0, dtype=np.float64)

#     feature = ((), '')  
#     assert feature in features
#     theta_series[feature] = np.float64(theta_0)

#     for i in range(L):
#         for j, c in enumerate(alphabet):
#             feature = ((i,), c)
#             assert feature in features
#             theta_series[feature] = theta_lc[i,j]
    
#     for i1 in range(L-1):
#         for i2 in range(i1+1,L):
#             for j1, c1 in enumerate(alphabet):
#                 for j2, c2 in enumerate(alphabet):
#                     feature = ((i1,i2), c1+c2)
#                     assert feature in features
#                     theta_series[feature] = theta_lclc[i1,j1,i2,j2]
#     return theta_series