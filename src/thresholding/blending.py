# src/thresholding/blending.py
from __future__ import annotations
from typing import Dict
import re
import numpy as np
import pandas as pd

from src.utils import naming as N
from src.thresholding.hybrid_labelling import hybrid_from_flags

# ---------- internals ----------

def _apply_flag(scores: pd.Series, thresh: pd.Series, direction: str) -> pd.Series:
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
    base = flag.fillna(False).astype(int).to_numpy()
    out = np.zeros_like(base, dtype=bool)
    run = 0
    for i, v in enumerate(base):
        run = run + 1 if v else 0
        out[i] = run >= k
    return pd.Series(out, index=flag.index, dtype="boolean")

def _extract_wq(thr_or_flag_col: str) -> tuple[int, int]:
    # matches model-scoped tokens like "__w1080_q030"
    m = re.search(r"__w(\d+)_q(\d{3})", thr_or_flag_col)
    if not m:
        raise ValueError(f"Cannot parse window/quantile token from column: {thr_or_flag_col}")
    return int(m.group(1)), int(m.group(2))

def _tag_from_pair(if_col: str, ae_col: str) -> str:
    w_if_h, q_if3 = _extract_wq(if_col)
    w_ae_h, q_ae3 = _extract_wq(ae_col)
    return N.tag(w_if_h, q_if3 / 1000.0, w_ae_h, q_ae3 / 1000.0)

# ---------- public API ----------

def capped_minmax_thresholds(
    df: pd.DataFrame,
    *,
    if_adaptive_thr_col: str,
    ae_adaptive_thr_col: str,
    static_if_thr: float,
    static_ae_thr: float,
    cap_delta: float,
) -> Dict[str, str]:
    """
    IF_blend = max( min(static_IF, IF_adapt), IF_adapt - δ )
    AE_blend = min( max(static_AE, AE_adapt), AE_adapt + δ )
    """
    # IF
    if_adapt = df[if_adaptive_thr_col]
    if_blend_vals = np.maximum(np.minimum(static_if_thr, if_adapt), if_adapt - cap_delta)
    if_blend_col = N.thr_if_blend(if_adaptive_thr_col, cap_delta)
    df[if_blend_col] = pd.Series(if_blend_vals, index=df.index)

    # AE
    ae_adapt = df[ae_adaptive_thr_col]
    ae_blend_vals = np.minimum(np.maximum(static_ae_thr, ae_adapt), ae_adapt + cap_delta)
    ae_blend_col = N.thr_if_blend(ae_adaptive_thr_col, cap_delta)  # same suffixing scheme
    df[ae_blend_col] = pd.Series(ae_blend_vals, index=df.index)

    return {"if_blend_thresh": if_blend_col, "ae_blend_thresh": ae_blend_col}

def flags_from_thresholds(
    df: pd.DataFrame,
    *,
    if_score_col: str,
    ae_score_col: str,
    if_thr_col: str,
    ae_thr_col: str,
) -> Dict[str, str]:
    """
    Build blended flags from the blended thresholds.
    Expects threshold names to contain "__w{hours}_q{qqq}" and end with "_blend_cap{ddd}".
    """
    # verify cap suffix matches
    m_if = re.search(r"_blend_cap(\d{3})$", if_thr_col)
    m_ae = re.search(r"_blend_cap(\d{3})$", ae_thr_col)
    if not (m_if and m_ae and m_if.group(1) == m_ae.group(1)):
        raise ValueError("Blend cap suffix mismatch between IF/AE thresholds.")
    cap = int(m_if.group(1)) / 1000.0

    # base adaptive flag column names (model-scoped tokens)
    w_if_h, q_if3 = _extract_wq(if_thr_col)
    w_ae_h, q_ae3 = _extract_wq(ae_thr_col)
    base_if_flag = N.flg_if_adapt(w_if_h, q_if3 / 1000.0)
    base_ae_flag = N.flg_ae_adapt(w_ae_h, q_ae3 / 1000.0)

    # blended flag column names
    if_flag_col = N.flg_if_blend(base_if_flag, cap)
    ae_flag_col = N.flg_if_blend(base_ae_flag, cap)

    # compute flags
    df[if_flag_col] = _apply_flag(df[if_score_col], df[if_thr_col], "low")
    df[ae_flag_col] = _apply_flag(df[ae_score_col], df[ae_thr_col], "high")

    return {"if_flag_col": if_flag_col, "ae_flag_col": ae_flag_col}

def dwell_on_pattern_only(
    df: pd.DataFrame,
    *,
    if_flag_col: str,
    ae_flag_col: str,
    k: int,
) -> Dict[str, str]:
    """
    Apply k-consecutive dwell to AE only (Pattern). IF (Point) unchanged.
    """
    if_dwell_col = if_flag_col
    ae_dwell_col = N.dwell(ae_flag_col, k)
    df[ae_dwell_col] = _dwell_runlength(df[ae_flag_col], k)

    tag = _tag_from_pair(if_flag_col, ae_flag_col)
    hybrid_col = N.hybrid_name(f"pattern_dwell{k}", tag)
    df[hybrid_col] = hybrid_from_flags(df[if_dwell_col].astype("boolean"),
                                       df[ae_dwell_col].astype("boolean"))

    return {"if_flag_col": if_dwell_col, "ae_dwell_col": ae_dwell_col, "hybrid_col": hybrid_col}
