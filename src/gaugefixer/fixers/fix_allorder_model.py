# Standard library imports
from itertools import product

# Third party imports
import numpy as np
import pandas as pd
from typeguard import typechecked

# Local imports
from gaugefixer.fixers._kron_matvec import _kron_matvec
from gaugefixer.features.petti_feature import PettiFeature
from gaugefixer.features import get_allorder_features

@typechecked
def fix_allorder_model(
    theta: pd.Series | np.ndarray,
    gauge: str | None,
    L: int,
    alphabet: list[str],
    features: list[PettiFeature] | None = None,
    wt_seq: str | None = None,
    lam: float | None = None,
    pi_lc: np.ndarray | None = None
    ) -> pd.Series:
    """Fix the gauge of parameters in an all-order interaction model.
    
    This function applies gauge fixing to parameters in an all-order interaction model.
    The gauge is specified either by a named gauge type or by providing lambda and pi_lc values directly.

    Args:
        theta (pd.Series | np.ndarray): Model parameters as either a pandas Series or numpy array
        gauge (str | None): Type of gauge fixing to apply. Options are:
            - 'wild-type': Fix parameters relative to a wild-type sequence
            - 'zero-sum': Zero-sum gauge fixing
            - 'hierarchical': Hierarchical gauge fixing with provided pi_lc
            - 'trivial': No gauge fixing (lambda=0)
            - 'euclidean': Euclidean gauge fixing (lambda=1)
            - 'equitable': Equitable gauge fixing (lambda=|alphabet|)
            - None: Custom gauge fixing with provided lambda and pi_lc
        L (int): Length of sequences
        alphabet (list[str]): List of characters in the sequence alphabet
        features (list[PettiFeature] | None): List of PettiFeatures if theta is a numpy array. If None, features are inferred.
        wt_seq (str | None): Wild-type sequence for wild-type gauge fixing
        lam (float | None): Custom lambda value for gauge fixing
        pi_lc (np.ndarray | None): Custom position-specific background frequencies (L x |alphabet| array)

    Returns:
        pd.Series: Gauge-fixed parameters indexed by features
    """
    alpha = len(alphabet)
    
    # Handle different gauge fixing cases
    match (gauge, lam, pi_lc, wt_seq):
        
        case ('wild-type', None, None, str()):
            assert len(wt_seq) == L
            assert set(wt_seq) <= set(alphabet)
            lam = np.inf
            pi_lc = np.array([[c==wt_c for j,c in enumerate(alphabet)] for i,wt_c in enumerate(wt_seq)])
            
        case ('zero-sum', None, None, None):
            lam = np.inf
            pi_lc = np.ones(shape=(L,alpha))/alpha
            
        case ('hierarchical', None, np.ndarray(), None):
            assert pi_lc.shape == (L, alpha)
            lam = np.inf
            
        case ('trivial', None, None, None):
            lam = 0
            pi_lc = np.ones(shape=(L,alpha))/alpha
            
        case ('euclidean', None, None, None):
            lam = 1
            pi_lc = np.ones(shape=(L,alpha))/alpha
            
        case ('equitable', None, None, None):
            lam = alpha
            pi_lc = np.ones(shape=(L,alpha))/alpha
            
        case (None, float(), np.ndarray(), None):
            pass
        
        case _:
            assert False, f'Invalid combination of inputs {gauge=}, {lam=}, {pi_lc=}, {wt_seq=}.'

    # Generate augmented sequences and features
    augalphabet = ['*']+alphabet
    augseqs = [''.join(seq) for seq in product(augalphabet, repeat=L)]
    features_in_kron_order = []
    for augseq in augseqs:
        orbit = tuple(i for i, c in enumerate(augseq) if c != '*')
        subseq = ''.join(c for c in augseq if c != '*')
        features_in_kron_order.append((orbit, subseq))
    
    # Convert input theta to pandas Series with proper features
    match (theta, features):
        
        case (pd.Series(), None):
            features = list(theta.index)
            assert set(features)==set(features_in_kron_order)
            theta_series = theta.copy()
            
        case (np.ndarray(), list()): 
            assert set(features)==set(features_in_kron_order)
            assert len(theta)==len(features)
            theta_series = pd.Series(data=theta, index=features)
            
        case (np.ndarray(), None):
            assert len(theta)==len(features_in_kron_order)
            features = get_allorder_features(L, alphabet) # Assume by default 
            theta_series = pd.Series(data=theta, index=features)
        
        case _:
            assert False, f'Invalid combination of inputs {theta=}, {features=}'
    
    # Calculate eta from lambda
    assert lam >= 0, "Lambda must be non-negative"
    if np.isfinite(lam):
        eta = lam/(1+lam)
    elif lam == np.inf:
        eta = 1.0
    else:
        raise ValueError(f"Invalid lambda value: {lam}")
    
    # Build factor matrices for Kronecker product
    factor_matrices = []
    for i in range(L):
        mat = np.zeros((alpha+1,alpha+1))
        mat[0,0] = eta
        mat[1:,0] = 1-eta
        mat[0,1:] = eta*pi_lc[i,:]
        mat[1:,1:] = np.eye(alpha) - eta*pi_lc[i,:][np.newaxis,:]
        factor_matrices.append(mat)
        
    # Apply gauge fixing transformation
    in_features = list(theta_series.index)
    theta_series = theta_series.reindex(features_in_kron_order)
    theta_fixed_values = _kron_matvec(factor_matrices, theta_series.values)
    theta_fixed_series = pd.Series(data=theta_fixed_values, index=features_in_kron_order)
    theta_fixed_series = theta_fixed_series.reindex(in_features)
    
    return theta_fixed_series 