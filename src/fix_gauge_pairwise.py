import numpy as np
import pandas as pd
import numbers

from src.pairwise_theta_series_to_dict import pairwise_theta_series_to_dict
from src.pairwise_theta_dict_to_series import pairwise_theta_dict_to_series

def fix_gauge_pairwise_theta_dict(theta_dict: dict, p_lc: np.ndarray) -> dict:

    # Extract variables from dict
    L = theta_dict['L']
    alphabet = theta_dict['alphabet']
    theta_0 = theta_dict['theta_0']
    theta_lc = theta_dict['theta_lc']
    theta_lclc = theta_dict['theta_lclc']
    
    # Useful alias
    _ = np.newaxis
    
    # Record nan masks and then set nan values to zero.
    nan_mask_lclc = np.isnan(theta_lclc)
    theta_lclc[nan_mask_lclc] = 0

    # Fix 0th order parameter
    fixed_theta_0 = theta_0 \
        + np.sum(p_lc * theta_lc) \
        + np.sum(theta_lclc * p_lc[:, :, _, _] * p_lc[_, _, :, :])

    # Fix 1st order parameters
    fixed_theta_lc = theta_lc \
        - np.sum(theta_lc * p_lc, axis=1)[:, _] \
        + np.sum(theta_lclc * p_lc[_, _, :, :],
                    axis=(2, 3)) \
        + np.sum(theta_lclc * p_lc[:, :, _, _],
                    axis=(0, 1)) \
        - np.sum(theta_lclc * p_lc[:, :, _, _] * p_lc[_, _, :, :],
                    axis=(1, 2, 3))[:, _] \
        - np.sum(theta_lclc * p_lc[:, :, _, _] * p_lc[_, _, :, :],
                    axis=(0, 1, 3))[:, _]

    # Fix 2nd order parameters
    fixed_theta_lclc = theta_lclc \
        - np.sum(theta_lclc * p_lc[:, :, _, _],
                    axis=1)[:, _, :, :] \
        - np.sum(theta_lclc * p_lc[_, _, :, :],
                    axis=3)[:, :, :, _] \
        + np.sum(theta_lclc * p_lc[:, :, _, _] * p_lc[_, _, :, :],
                    axis=(1, 3))[:, _, :, _]

    # Set and return output
    fixed_theta_dict = {
        'L': L,
        'alphabet': alphabet,
        'theta_0': fixed_theta_0,
        'theta_lc': fixed_theta_lc,
        'theta_lclc': fixed_theta_lclc,
    }

    return fixed_theta_dict


def fix_gauge_pairwise_theta_series(theta_series: pd.Series, p_lc: np.ndarray, alphabet: str, L: int) -> pd.Series:
    # Convert series to dict
    theta_dict = pairwise_theta_series_to_dict(theta_series, alphabet=alphabet, L=L)

    # Fix gauge
    theta_fixed_dict = fix_gauge_pairwise_theta_dict(theta_dict, p_lc=p_lc)

    # Convert back to series
    theta_fixed_series = pairwise_theta_dict_to_series(theta_fixed_dict)

    return theta_fixed_series
