# src/thresholding/rolling.py
from __future__ import annotations

from typing import Literal, Tuple, Optional
import pandas as pd

from src.utils import naming as N

Direction = Literal["low", "high"]


def make_colnames(
    model_prefix: Literal["if", "lstm"],
    window: int,
    quantile: float,
    combo_tag: str,
) -> Tuple[str, str]:
    """
    Return (threshold_col_name, flag_col_name) using canonical naming helpers.
    """
    if model_prefix == "if":
        thresh_col = N.thr_if_adapt(window, quantile, combo_tag)
        flag_col = N.flg_if_adapt(window, quantile, combo_tag)
    elif model_prefix == "lstm":
        thresh_col = N.thr_ae_adapt(window, quantile, combo_tag)
        flag_col = N.flg_ae_adapt(window, quantile, combo_tag)
    else:
        raise ValueError(f"Unknown model_prefix: {model_prefix}")
    return thresh_col, flag_col


def compute_rolling_threshold(
    df: pd.DataFrame,
    score_col: str,
    window: int,
    quantile: float,
    model_prefix: Literal["if", "lstm"],
    combo_tag: str,
    min_periods: Optional[int] = None,
) -> str:
    """
    Compute a trailing rolling quantile threshold over `score_col` and store it in-place.
    """
    if min_periods is None:
        min_periods = window

    thresh_col, _ = make_colnames(model_prefix, window, quantile, combo_tag)

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
    """
    Apply thresholding to create a nullable-boolean flag column in-place.
    """
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
    combo_tag: str,
    direction: Direction,
    min_periods: Optional[int] = None,
) -> Tuple[str, str]:
    """
    Convenience wrapper: compute the rolling threshold and its flag in one call.
    Returns (threshold_col, flag_col).
    """
    thresh_col, flag_col = make_colnames(model_prefix, window, quantile, combo_tag)
    compute_rolling_threshold(df, score_col, window, quantile, model_prefix, combo_tag, min_periods)
    apply_flag(df, score_col, thresh_col, direction, flag_col)
    return thresh_col, flag_col
