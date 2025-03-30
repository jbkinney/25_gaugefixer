import numpy as np

def pairwise_theta_series_to_dict(theta_series, alphabet, wildcard='*'):
    L = len(theta_series.index[0])
    features = theta_series.index
    alpha = len(alphabet)

    # Goal: transform features to theat_0, theta_lc, theta_lclc. 
    theta_0 = np.float64(0)
    theta_lc = np.zeros((L,alpha), dtype=np.float64)
    theta_lclc = np.zeros((L,alpha,L,alpha),  dtype=np.float64)

    feature = wildcard*L
    assert feature in features
    theta_0 = theta_series[feature]

    for i in range(L):
        for j, c in enumerate(alphabet):
            feature = wildcard*i + c + wildcard*(L-i-1)
            assert feature in features
            theta_lc[i,j] = theta_series[feature]
    
    for i1 in range(L):
        for i2 in range(i1+1,L):
            for j1, c1 in enumerate(alphabet):
                for j2, c2 in enumerate(alphabet):
                    feature = wildcard*i1 + c1 + wildcard*(i2-i1-1) + c2 + wildcard*(L-i2-1)
                    assert feature in features
                    theta_lclc[i1,j1,i2,j2] = theta_series[feature]
                
    # Store as dict
    return dict(theta_0=theta_0, theta_lc=theta_lc, theta_lclc=theta_lclc, L=L, alpha=alpha, alphabet=alphabet, wildcard=wildcard)
