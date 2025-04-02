from src._pairwise_series2dict import _pairwise_series2dict
from src._pairwise_dict2series import _pairwise_dict2series
from src._fix_pairwise_dict import fix_hg_pairwise_dict
import numpy as np
import pandas as pd
from typeguard import typechecked

@typechecked
def fix_hg_pairwise_series(theta_series: pd.Series, p_lc: np.ndarray, alphabet: list[str], L: int) -> pd.Series:
    # Convert series to dict
    theta_dict = _pairwise_series2dict(theta_series, alphabet=alphabet, L=L)

    # Fix gauge
    theta_fixed_dict = fix_hg_pairwise_dict(theta_dict, p_lc=p_lc)

    # Convert back to series
    theta_fixed_series = _pairwise_dict2series(theta_fixed_dict)

    return theta_fixed_series