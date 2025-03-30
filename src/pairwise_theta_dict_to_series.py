import pandas as pd
import numpy as np
from src import get_pairwise_model_features

def pairwise_theta_dict_to_series(theta_dict):
    wildcard = theta_dict['wildcard']
    L = theta_dict['L']
    alphabet = theta_dict['alphabet']
    alpha = len(alphabet)
    theta_0 = theta_dict['theta_0']
    theta_lc = theta_dict['theta_lc']
    theta_lclc = theta_dict['theta_lclc']
    
    # Make features
    features = get_pairwise_model_features(L=L, alphabet=alphabet, wildcard_char=wildcard)
    
    # Create a lookup dictionary relating features to their indices
    feature_to_ix_dict = {seq: idx for idx, seq in enumerate(features)}
    
    # Initialize theta_series
    theta_series = pd.Series(index=features, data=0, dtype=np.float64)
    
    feature = wildcard*L
    assert feature in features
    theta_series[feature] = np.float64(theta_0)

    # Get first character
    c0 = alphabet[0]

    for i in range(L):
        feature = wildcard*i + c0 + wildcard*(L-i-1)
        assert feature in features
        ix = feature_to_ix_dict[feature]
        theta_series.iloc[ix:ix+alpha] = theta_lc[i,:]
    
    for i1 in range(L):
        for i2 in range(i1+1,L):
            for j2, c2 in enumerate(alphabet):
                feature = wildcard*i1 + c0 + wildcard*(i2-i1-1) + c2 + wildcard*(L-i2-1)
                assert feature in features
                ix = feature_to_ix_dict[feature]
                theta_series.iloc[ix:ix+alpha] = theta_lclc[i1,:,i2,j2]
                    
    return theta_series


# def pairwise_theta_dict_to_series(theta_dict):
#     wildcard = theta_dict['wildcard']
#     L = theta_dict['L']
#     alphabet = theta_dict['alphabet']
#     theta_0 = theta_dict['theta_0']
#     theta_lc = theta_dict['theta_lc']
#     theta_lclc = theta_dict['theta_lclc']
    
#     # Make features
#     features = get_pairwise_model_features(L=L, alphabet=alphabet, wildcard_char=wildcard)
    
#     # Create a lookup dictionary relating features to their indices
#     features_to_index = {seq: idx for idx, seq in enumerate(features)}
    
#     # Initialize theta_series
#     theta_series = pd.Series(index=features, data=0, dtype=np.float64)
    
#     feature = wildcard*L
#     assert feature in features
#     theta_series[feature] = np.float64(theta_0)

#     for i in range(L):
#         for j, c in enumerate(alphabet):
#             feature = wildcard*i + c + wildcard*(L-i-1)
#             assert feature in features
#             theta_series[feature] = theta_lc[i,j]
    
#     for i1 in range(L):
#         for i2 in range(i1+1,L):
#             for j1, c1 in enumerate(alphabet):
#                 for j2, c2 in enumerate(alphabet):
#                     feature = wildcard*i1 + c1 + wildcard*(i2-i1-1) + c2 + wildcard*(L-i2-1)
#                     assert feature in features
#                     theta_series[feature] = theta_lclc[i1,j1,i2,j2]
                    
#     return theta_series
