# Standard library imports
from itertools import product

# Third party imports
import numpy as np
import pandas as pd
from typeguard import typechecked

# Local imports
from gaugefixer.features.petti_feature import PettiFeature
from gaugefixer.features import get_additive_features
from gaugefixer.fixers._additive_series2dict import _additive_series2dict
from gaugefixer.fixers._additive_dict2series import _additive_dict2series
from gaugefixer.fixers._fix_additive_dict import _fix_additive_dict

@typechecked
def fix_additive_model(
    theta: pd.Series | np.ndarray,
    gauge: str | None,
    L: int,
    alphabet: list[str],
    features: list[PettiFeature] | None = None,
    wt_seq: str | None = None,
    pi_lc: np.ndarray | None = None
    ) -> pd.Series:
    """Fix the gauge of parameters in an additive model.
    
    This function applies gauge fixing to parameters in an additive model.
    The gauge can be specified either as 'wild-type', 'zero-sum', or by providing custom
    background frequencies pi_lc directly.

    Args:
        theta (pd.Series | np.ndarray): Model parameters as either a pandas Series indexed by features
            or a numpy array that will be matched with provided features
        gauge (str | None): Type of gauge fixing to apply:
            - 'wild-type': Fix parameters relative to a wild-type sequence
            - 'zero-sum': Use uniform background frequencies (1/|alphabet|)
            - None: Use custom background frequencies provided in pi_lc
        L (int): Length of sequences
        alphabet (list[str]): List of characters in the sequence alphabet
        features (list[PettiFeature] | None): List of features if theta is a numpy array. 
            If None and theta is a numpy array, features are inferred from L and alphabet
        wt_seq (str | None): Wild-type sequence for wild-type gauge fixing. Required if gauge='wild-type'
        pi_lc (np.ndarray | None): Custom position-specific background frequencies as L x |alphabet| array.
            Required if gauge=None

    Returns:
        pd.Series: Gauge-fixed parameters indexed by the same features as the input theta

    Raises:
        AssertionError: If invalid combination of gauge, pi_lc and wt_seq is provided,
            or if features don't match the expected additive features for sequence length L
    """
    alpha = len(alphabet)
    
    # Handle different gauge fixing cases
    match (gauge, pi_lc, wt_seq):
        
        case ('wild-type', None, str()):
            assert len(wt_seq) == L
            assert set(wt_seq) <= set(alphabet)
            pi_lc = np.array([[c==wt_c for j,c in enumerate(alphabet)] for i,wt_c in enumerate(wt_seq)])
            
        case ('zero-sum', None, None):
            pi_lc = np.ones(shape=(L,alpha))/alpha
            
        case (None, np.ndarray(), None):
            assert pi_lc.shape==(L,alpha)
        
        case _:
            assert False, f'Invalid combination of inputs {gauge=}, {pi_lc=}, {wt_seq=}.'

    # Generate augmented sequences and features
    sorted_features = get_additive_features(L, alphabet)
    
    # Convert input theta to pandas Series with proper features
    match (theta, features):
        
        case (pd.Series(), None):
            assert set(theta.index)==set(sorted_features)
            theta_series = theta.copy() 
            
        case (np.ndarray(), list()): 
            assert set(features)==set(sorted_features)
            assert len(theta)==len(features)
            theta_series = pd.DataFrame(data=theta, index=features)
            
        case (np.ndarray(), None):
            assert len(theta)==len(sorted_features)
            theta_series = pd.DataFrame(data=theta, index=sorted_features)
        
        case _:
            assert False, f'Invalid combination of inputs {theta=}, {features=}'
    
    
    # Save in_features
    in_features = list(theta_series.index)
    
    # Sort theta_series according to sorted_features
    theta_series = theta_series.loc[sorted_features]
    
    # Convert series to dict
    theta_dict = _additive_series2dict(theta_series, alphabet=alphabet, L=L)

    # Fix gauge
    theta_fixed_dict = _fix_additive_dict(theta_dict, p_lc=pi_lc)

    # Convert back to series
    theta_fixed_series = _additive_dict2series(theta_fixed_dict)
    
    # Reorder fixed thetas according to in_features    
    theta_fixed_series = theta_fixed_series.loc[in_features]
    
    return theta_fixed_series 