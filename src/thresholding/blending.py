import numpy as np
import pandas as pd
from src.thresholding.hybrid_labelling import hybrid_from_flags
# -----------------------------
# Low-level helpers
# -----------------------------

def _apply_flag(scores: pd.Series, thresh: pd.Series, direction: str) -> pd.Series:
    """
    Vectorised thresholding with nullable boolean output.
    direction: 'low' => score <= threshold (IF)
               'high' => score >= threshold (AE)
    """
    out = pd.Series(pd.NA, index=scores.index, dtype="boolean")
    valid = thresh.notna()
    if direction == "low":
        out.loc[valid] = (scores.loc[valid] <= thresh.loc[valid]).astype("boolean")
    elif direction == "high":
        out.loc[valid] = (scores.loc[valid] >= thresh.loc[valid]).astype("boolean")
    else:
        raise ValueError("direction must be 'low' or 'high'")
    return out

def _dwell_runlength(flag: pd.Series, k: int) -> pd.Series:
    """
    Emit True after >=k consecutive True; NA treated as False.
    """
    base = flag.fillna(False).astype(int).to_numpy()
    out = np.zeros_like(base, dtype=bool)
    run = 0
    for i, v in enumerate(base):
        run = run + 1 if v else 0
        out[i] = run >= k
    return pd.Series(out, index=flag.index, dtype="boolean")

# -----------------------------
# Public API
# -----------------------------

def capped_minmax_thresholds(
    df: pd.DataFrame,
    if_adaptive_thr_col: str,
    ae_adaptive_thr_col: str,
    static_if_thr: float,
    static_ae_thr: float,
    cap_delta: float,
) -> dict:
    """
    Apply capped min–max blending to adaptive thresholds.

    Notes
    -----
    - Input adaptive thresholds are already computed over rolling WINDOWS (in HOURS).
    - This function enforces:
        IF_blend = max( min(static_IF, IF_adapt), IF_adapt - δ )
        AE_blend = min( max(static_AE, AE_adapt), AE_adapt + δ )
    - Resulting thresholds are stored as new DataFrame columns.
    """
    # IF
    if_adapt = df[if_adaptive_thr_col]
    if_blend = np.maximum(np.minimum(static_if_thr, if_adapt), if_adapt - cap_delta)
    if_blend_col = f"{if_adaptive_thr_col}_blend_cap{int(cap_delta*1000):03d}"
    df[if_blend_col] = pd.Series(if_blend, index=df.index)

    # AE
    ae_adapt = df[ae_adaptive_thr_col]
    ae_blend = np.minimum(np.maximum(static_ae_thr, ae_adapt), ae_adapt + cap_delta)
    ae_blend_col = f"{ae_adaptive_thr_col}_blend_cap{int(cap_delta*1000):03d}"
    df[ae_blend_col] = pd.Series(ae_blend, index=df.index)

    return {
        "if_blend_thresh": if_blend_col,
        "ae_blend_thresh": ae_blend_col,
    }

def flags_from_thresholds(
    df: pd.DataFrame,
    if_score_col: str,
    ae_score_col: str,
    if_thr_col: str,
    ae_thr_col: str,
) -> dict:
    """
    Derive nullable-boolean flags from thresholds.
    Uses IF 'low' and AE 'high' directions.
    """
    if_flag_col = if_thr_col.replace("adaptive_thresh", "adaptive_flag").replace("_blend_", "_blend_flag_")
    ae_flag_col = ae_thr_col.replace("adaptive_thresh", "adaptive_flag").replace("_blend_", "_blend_flag_")
    df[if_flag_col] = _apply_flag(df[if_score_col], df[if_thr_col], "low")
    df[ae_flag_col] = _apply_flag(df[ae_score_col], df[ae_thr_col], "high")
    return {"if_flag_col": if_flag_col, "ae_flag_col": ae_flag_col}

def dwell_on_pattern_only(df, if_flag_col: str, ae_flag_col: str, k: int) -> dict:
    """
    Apply k-consecutive True dwell ONLY to AE (Pattern) anomalies.
    IF (Point) anomalies are left unchanged.
    """
    # Keep IF unchanged
    if_dwell_col = if_flag_col

    # Apply dwell to AE only
    ae_dwell_col = f"{ae_flag_col}_dwell{k}"
    df[ae_dwell_col] = _dwell_runlength(df[ae_flag_col], k)

    # Build hybrid
    hybrid_col = f"hybrid_label_pattern_dwell{k}__{if_flag_col.split('_w')[1]}__{ae_flag_col.split('_w')[1]}"
    df[hybrid_col] = hybrid_from_flags(df[if_flag_col], df[ae_dwell_col])

    return {
        "if_flag_col": if_dwell_col,
        "ae_dwell_col": ae_dwell_col,
        "hybrid_col": hybrid_col,
    }

def dwell_on_flags(
    df: pd.DataFrame,
    if_flag_col: str,
    ae_flag_col: str,
    k: int,
) -> dict:
    """
    Apply k-consecutive True dwell to IF/AE flag columns.
    """
    if_dwell_col = f"{if_flag_col}_dwell{k}"
    ae_dwell_col = f"{ae_flag_col}_dwell{k}"
    df[if_dwell_col] = _dwell_runlength(df[if_flag_col], k)
    df[ae_dwell_col] = _dwell_runlength(df[ae_flag_col], k)
    return {"if_dwell_col": if_dwell_col, "ae_dwell_col": ae_dwell_col}

