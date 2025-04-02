from src._additive_series2dict import _additive_series2dict
from src._additive_dict2series import _additive_dict2series
from src._fix_additive_dict import _fix_additive_dict
import numpy as np
import pandas as pd
from typeguard import typechecked

@typechecked
def fix_hg_additive_series(theta_series: pd.Series, p_lc: np.ndarray, alphabet: list[str], L: int) -> pd.Series:
    # Convert series to dict
    theta_dict = _additive_series2dict(theta_series, alphabet=alphabet, L=L)

    # Fix gauge
    theta_fixed_dict = _fix_additive_dict(theta_dict, p_lc=p_lc)

    # Convert back to series
    theta_fixed_series = _additive_dict2series(theta_fixed_dict)

    return theta_fixed_series