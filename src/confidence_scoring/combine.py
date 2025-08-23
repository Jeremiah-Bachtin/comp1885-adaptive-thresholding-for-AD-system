# src/confidence_scoring/combine.py
from __future__ import annotations
import numpy as np
import pandas as pd

def _safe_mean(a, b):
    vals = [v for v in (a, b) if v is not None and not np.isnan(v)]
    if not vals:
        return np.nan
    return float(np.mean(vals))

def combine_by_label(label: str | None, if_conf: float | None, ae_conf: float | None, *,
                     point_source="IF", pattern_source="AE", compound_agg="mean") -> float | None:
    if label is None:
        return np.nan

    if label == "Point":
        return float(if_conf) if if_conf is not None else np.nan
    if label == "Pattern":
        return float(ae_conf) if ae_conf is not None else np.nan
    if label == "Compound":
        if compound_agg == "mean":
            return _safe_mean(if_conf, ae_conf)
        # hooks for min/max/harmonic could go here
        return _safe_mean(if_conf, ae_conf)
    # label == <NA>/None or "None"
    return np.nan

def _to_float_or_nan(x):
    if x is None:
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan

def _mean_safe(a, b) -> float:
    """
    Mean of available confidences; returns NaN if both are missing.
    """
    vals = [_to_float_or_nan(a), _to_float_or_nan(b)]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else np.nan

def combine_by_label(label: str, if_conf: float | None, ae_conf: float | None) -> float:
    """
    Label-aware combination:
      - Point   -> IF confidence (NaN if missing)
      - Pattern -> AE confidence (NaN if missing)
      - Compound-> mean(IF, AE) over available inputs (NaN if both missing)
      - None/other -> NaN
    """
    lab = (label or "").strip().title()
    if lab == "Point":
        return _to_float_or_nan(if_conf)
    if lab == "Pattern":
        return _to_float_or_nan(ae_conf)
    if lab == "Compound":
        return _mean_safe(if_conf, ae_conf)
    return np.nan

def emit_confidence_column(
    df: pd.DataFrame,
    *,
    label_col: str,
    if_conf_col: str,
    ae_conf_col: str,
    out_col: str,
    emit_types: set[str],
) -> pd.DataFrame:
    """
    Build a single confidence column per timestamp. Rows whose label is not in
    `emit_types` are set to NaN without attempting a combine.
    """
    out = df.copy()
    labels = out[label_col].astype("string")
    types = {t.title() for t in emit_types}

    IF = out[if_conf_col] if if_conf_col in out else pd.Series(np.nan, index=out.index)
    AE = out[ae_conf_col] if ae_conf_col in out else pd.Series(np.nan, index=out.index)

    vals = []
    for i, lab in enumerate(labels):
        if lab not in types:
            vals.append(np.nan)
            continue
        vals.append(
            combine_by_label(
                lab,
                None if pd.isna(IF.iat[i]) else float(IF.iat[i]),
                None if pd.isna(AE.iat[i]) else float(AE.iat[i]),
            )
        )

    out[out_col] = vals
    return out
