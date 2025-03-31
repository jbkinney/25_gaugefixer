from src.switch_pairwise_series2dict import switch_pairwise_theta_series_to_dict
from src.switch_pairwise_dict2series import switch_pairwise_theta_dict2series
from src.fix_hg_pairwise_dict import fix_hg_pairwise_dict
import numpy as np
import pandas as pd
from typeguard import typechecked

@typechecked
def fix_hg_pairwise_series(theta_series: pd.Series, p_lc: np.ndarray, alphabet: list[str], L: int) -> pd.Series:
    # Convert series to dict
    theta_dict = switch_pairwise_theta_series_to_dict(theta_series, alphabet=alphabet, L=L)

    # Fix gauge
    theta_fixed_dict = fix_hg_pairwise_dict(theta_dict, p_lc=p_lc)

    # Convert back to series
    theta_fixed_series = switch_pairwise_theta_dict2series(theta_fixed_dict)

    return theta_fixed_series