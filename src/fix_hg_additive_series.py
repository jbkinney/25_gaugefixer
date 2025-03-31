from src.switch_additive_series2dict import switch_additive_series2dict
from src.swight_additive_dict2series import switch_additive_dict2series
from src.fix_hg_additive_dict import fix_hg_additive_dict
import numpy as np
import pandas as pd
from typeguard import typechecked

@typechecked
def fix_hg_additive_series(theta_series: pd.Series, p_lc: np.ndarray, alphabet: list[str], L: int) -> pd.Series:
    # Convert series to dict
    theta_dict = switch_additive_series2dict(theta_series, alphabet=alphabet, L=L)

    # Fix gauge
    theta_fixed_dict = fix_hg_additive_dict(theta_dict, p_lc=p_lc)

    # Convert back to series
    theta_fixed_series = switch_additive_dict2series(theta_fixed_dict)

    return theta_fixed_series