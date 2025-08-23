import pandas as pd

from .ids import q_to_str

def make_colnames(model_prefix: str, window: int, quantile: float):
    """
    Returns (threshold_col_name, flag_col_name) using the 3-digit quantile string.
    model_prefix: 'if' or 'lstm'
    """
    qstr = q_to_str(quantile)
    thresh_col = f"{model_prefix}_adaptive_thresh_w{window}_q{qstr}"
    flag_col   = f"is_{model_prefix}_adaptive_w{window}_q{qstr}"
    return thresh_col, flag_col


def compute_rolling_threshold(df: pd.DataFrame, score_col: str,
                              window: int, quantile: float,
                              model_prefix: str, min_periods: int|None=None) -> str:
    if min_periods is None:
        min_periods = window
    thresh_col, _ = make_colnames(model_prefix, window, quantile)
    df[thresh_col] = (
        df[score_col]
        .rolling(window=window, min_periods=min_periods)
        .quantile(quantile)
    )
    return thresh_col

def apply_flag(df, score_col, threshold_col, direction, flag_col):
    if direction == "low":
        cond = df[score_col] <= df[threshold_col]
    elif direction == "high":
        cond = df[score_col] >= df[threshold_col]
    else:
        raise ValueError("direction must be 'low' or 'high'")

    valid = df[threshold_col].notna()
    # nullable boolean so it can hold <NA>
    df[flag_col] = pd.Series(pd.NA, index=df.index, dtype="boolean")
    df.loc[valid, flag_col] = cond[valid].astype("boolean")
    return flag_col

def compute_and_flag(df: pd.DataFrame, score_col: str,
                     window: int, quantile: float, model_prefix: str,
                     direction: str, min_periods: int|None=None):
    """One call that computes the rolling quantile AND the boolean flag."""
    thresh_col, flag_col = make_colnames(model_prefix, window, quantile)
    compute_rolling_threshold(df, score_col, window, quantile, model_prefix, min_periods)
    apply_flag(df, score_col, thresh_col, direction, flag_col)
    return thresh_col, flag_col


