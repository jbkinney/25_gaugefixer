import numpy as np
from typeguard import typechecked

@typechecked
def fix_hg_additive_dict(theta_dict: dict, p_lc: np.ndarray) -> dict:

    # Extract variables from dict
    L = theta_dict['L']
    alphabet = theta_dict['alphabet']
    theta_0 = theta_dict['theta_0']
    theta_lc = theta_dict['theta_lc']
    
    # Useful alias
    _ = np.newaxis
    
    # Fix 0th order parameter
    fixed_theta_0 = theta_0 \
        + np.sum(p_lc * theta_lc)

    # Fix 1st order parameters
    fixed_theta_lc = theta_lc \
        - np.sum(theta_lc * p_lc, axis=1)[:, _]

    # Set and return output
    fixed_theta_dict = {
        'L': L,
        'alphabet': alphabet,
        'theta_0': fixed_theta_0,
        'theta_lc': fixed_theta_lc,
    }

    return fixed_theta_dict



