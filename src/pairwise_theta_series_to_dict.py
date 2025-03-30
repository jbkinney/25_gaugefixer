import numpy as np
import pandas as pd

def pairwise_theta_series_to_dict(theta_series:pd.Series, alphabet:list[str], L:int):
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
    theta_series[feature] = np.float64(theta_0)

    # Set 1st order parameters
    for i in range(L):
        feature0 = ((i,), c0)
        assert feature0 in features
        ix = features_to_ix_dict[feature0]
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


# def pairwise_theta_series_to_dict(theta_series:pd.Series, alphabet:list[str], L:int):
#     features = theta_series.index
#     alpha = len(alphabet)

#     # Goal: transform features to theat_0, theta_lc, theta_lclc. 
#     theta_0 = np.float64(0)
#     theta_lc = np.zeros((L,alpha), dtype=np.float64)
#     theta_lclc = np.zeros((L,alpha,L,alpha),  dtype=np.float64)

#     feature = ((), '')  
#     assert feature in features
#     theta_0 = theta_series[feature]

#     for i in range(L):
#         for j, c in enumerate(alphabet):
#             feature = ((i,), c)
#             assert feature in features
#             theta_lc[i,j] = theta_series[feature]
    
#     for i1 in range(L):
#         for i2 in range(i1+1,L):
#             for j1, c1 in enumerate(alphabet):
#                 for j2, c2 in enumerate(alphabet):
#                     feature = ((i1,i2), c1+c2)
#                     assert feature in features
#                     theta_lclc[i1,j1,i2,j2] = theta_series[feature]
                
#     # Store as dict
#     return dict(theta_0=theta_0, theta_lc=theta_lc, theta_lclc=theta_lclc, L=L, alpha=alpha, alphabet=alphabet)
