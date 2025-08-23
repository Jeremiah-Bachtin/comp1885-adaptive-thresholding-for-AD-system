from __future__ import annotations
from typing import Literal, Tuple, Optional
import pandas as pd

# use your helpers
from src.utils import naming as N

Direction = Literal["low", "high"]


def make_colnames(model_prefix: Literal["if", "lstm"], window: int, quantile: float) -> Tuple[str, str]:
    """
    Return (threshold_col_name, flag_col_name) using canonical helpers.

    model_prefix: 'if' or 'lstm' ; window is HOURS ; quantile in [0,1]
    """
    if model_prefix == "if":
        return N.thr_if_adapt(window, quantile), N.flg_if_adapt(window, quantile)
    elif model_prefix == "lstm":
        return N.thr_ae_adapt(window, quantile), N.flg_ae_adapt(window, quantile)
    raise ValueError("model_prefix must be 'if' or 'lstm'")


def compute_rolling_threshold(
    df: pd.DataFrame,
    score_col: str,
    window: int,
    quantile: float,
    model_prefix: Literal["if", "lstm"],
    min_periods: Optional[int] = None,
) -> str:
    if min_periods is None:
        min_periods = window
    thresh_col, _ = make_colnames(model_prefix, window, quantile)
    df[thresh_col] = (
        df[score_col]
        .rolling(window=window, min_periods=min_periods)
        .quantile(quantile)
    )
    return thresh_col


def apply_flag(
    df: pd.DataFrame,
    score_col: str,
    threshold_col: str,
    direction: Direction,
    flag_col: str,
) -> str:
    if direction == "low":
        cond = df[score_col] <= df[threshold_col]
    elif direction == "high":
        cond = df[score_col] >= df[threshold_col]
    else:
        raise ValueError("direction must be 'low' or 'high'")

    valid = df[threshold_col].notna()
    df[flag_col] = pd.Series(pd.NA, index=df.index, dtype="boolean")
    df.loc[valid, flag_col] = cond.loc[valid].astype("boolean")
    return flag_col


def compute_and_flag(
    df: pd.DataFrame,
    score_col: str,
    window: int,
    quantile: float,
    model_prefix: Literal["if", "lstm"],
    direction: Direction,
    min_periods: Optional[int] = None,
) -> Tuple[str, str]:
    thresh_col, flag_col = make_colnames(model_prefix, window, quantile)
    compute_rolling_threshold(df, score_col, window, quantile, model_prefix, min_periods)
    apply_flag(df, score_col, thresh_col, direction, flag_col)
    return thresh_col, flag_col
