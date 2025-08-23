# src/utils/data_utils.py
import os
import pandas as pd
from typing import Iterable, List

def load_scores(path: str) -> pd.DataFrame:
    """
    Load the scores dataset from a CSV file, sort by 'timestamp',
    and return a DataFrame.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If 'timestamp' column is missing or cannot be parsed.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Check your .env DATA_PATH setting. If relative, it should be "
            "relative to the project root, or use an absolute path."
        )

    df = pd.read_csv(path, parse_dates=["timestamp"])
    if "timestamp" not in df.columns or not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        raise ValueError("Column 'timestamp' must exist and be parseable as datetime.")
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def slice_by_date(df: pd.DataFrame, timestamp_col: str, start: str, end: str) -> pd.DataFrame:
    """
    Inclusive slice [start, end]. Expects ISO date strings (YYYY-MM-DD).
    Assumes `df[timestamp_col]` is a pandas datetime dtype.
    """
    start_ts = pd.to_datetime(start)
    end_ts   = pd.to_datetime(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    m = (df[timestamp_col] >= start_ts) & (df[timestamp_col] <= end_ts)
    return df.loc[m].copy()


def mask_valid_rows(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    """
    Return a boolean mask where all of the given columns are non-null.
    """
    return df[cols].notna().all(axis=1)


def require_columns(df: pd.DataFrame, cols: Iterable[str], context: str = "") -> None:
    """
    Raise a clear error if any of the required columns are missing.

    Parameters
    ----------
    df : DataFrame
    cols : iterable of column names
    context : str, optional description to include in the error message.
    """
    cols = list(cols)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        ctx = f" in {context}" if context else ""
        raise ValueError(f"Missing required columns{ctx}: {missing}. Present: {list(df.columns)}")


def ensure_nullable_bools(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """
    Ensure provided columns use pandas' nullable boolean dtype.
    Converts in-place and returns the df for chaining.
    """
    for c in cols:
        if c in df.columns and df[c].dtype != "boolean":
            df[c] = df[c].astype("boolean")
    return df


def write_csv(df: pd.DataFrame, path: str, log_prefix: str = "[save]") -> None:
    """
    Create parent directory if needed and write CSV (index=False).
    Prints a short, consistent log line.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"{log_prefix} Wrote {len(df):,} rows -> {path}")

def ensure_static_flags(
    df: pd.DataFrame,
    *,
    if_score_col: str,
    ae_score_col: str,
    static_if_thr: float,
    static_ae_thr: float,
    static_if_col: str,
    static_ae_col: str,
    force_recreate: bool = False,
) -> list[str]:
    """
    Ensure static baseline anomaly flags exist on a DataFrame, using pandas'
    nullable boolean dtype (True/False/<NA>).

    Parameters
    ----------
    df : pd.DataFrame
        The scores data frame (must contain the score columns).
    if_score_col : str
        Column with Isolation Forest anomaly scores (lower = more anomalous).
    ae_score_col : str
        Column with Autoencoder/LSTM-AE reconstruction scores (higher = more anomalous).
    static_if_thr : float
        Fixed threshold for IF (rows with score <= threshold are flagged True).
    static_ae_thr : float
        Fixed threshold for AE (rows with score >= threshold are flagged True).
    static_if_col : str
        Output column name for IF static flag.
    static_ae_col : str
        Output column name for AE static flag.
    force_recreate : bool, default False
        If True, (re)compute the flags even if the columns already exist.

    Returns
    -------
    created : list[str]
        Names of columns that were created/recomputed.

    Notes
    -----
    - This function does not modify thresholds; it only (re)computes flags.
    - Enforces pandas' nullable boolean dtype so downstream code can rely on it.
    - IF direction is 'low' (<= threshold), AE direction is 'high' (>= threshold).
    """
    created: list[str] = []

    # IF static flag
    if force_recreate or (static_if_col not in df.columns):
        df[static_if_col] = (df[if_score_col] <= static_if_thr).astype("boolean")
        created.append(static_if_col)
    else:
        # ensure dtype
        if df[static_if_col].dtype != "boolean":
            df[static_if_col] = df[static_if_col].astype("boolean")

    # AE static flag
    if force_recreate or (static_ae_col not in df.columns):
        df[static_ae_col] = (df[ae_score_col] >= static_ae_thr).astype("boolean")
        created.append(static_ae_col)
    else:
        if df[static_ae_col].dtype != "boolean":
            df[static_ae_col] = df[static_ae_col].astype("boolean")

    return created

