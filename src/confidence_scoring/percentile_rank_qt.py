#src/confidence_scoring/percentile_rank_qt.py

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Literal, Optional

Tail = Literal["low", "high"]

def _ecdf_hazen(x: float, window_vals: np.ndarray) -> float:
    """
    Hazen plotting-position ECDF in [0,1], i.e. (rank - 0.5)/n.
    Avoids hard 0/1 at finite n and is smoother with ties.
    """
    w = window_vals[~np.isnan(window_vals)]
    n = w.size
    if n == 0:
        return np.nan
    w_sorted = np.sort(w)
    k = np.searchsorted(w_sorted, x, side="right")
    return (k - 0.5) / n

def rolling_percentile_conf(
    scores: pd.Series,
    *,
    tail: Tail,
    window_hours: int,
    min_periods: int,
    eps: float = 1e-9,
    clip_min: float = 1e-3,
    clip_max: float = 0.999,
    # existing fixed quantile (combo q) remains supported as a fallback
    threshold_q: Optional[float] = None,
    # NEW: realised threshold in score space, aligned to scores.index
    threshold_series: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Percentile-rank confidence over a trailing window that EXCLUDES the current row.
    If `threshold_series` is provided, we compute q_t = ECDF(theta_t) from the same
    trailing window and use q_t instead of fixed `threshold_q`.
    """
    s = scores.astype(float)
    s_prev = s.shift(1)  # exclude current to avoid look-ahead
    out = np.full(len(s), np.nan, dtype=float)

    vals = s.to_numpy()
    prev = s_prev.to_numpy()
    use_series = threshold_series is not None
    theta = threshold_series.astype(float).to_numpy() if use_series else None
    use_fixed = (threshold_q is not None)

    for i in range(len(vals)):
        start = max(0, i - window_hours)
        win = prev[start:i]
        if (i - start) < min_periods:
            continue

        F = _ecdf_hazen(vals[i], win)
        if np.isnan(F):
            continue

        # Determine reference quantile q*
        if use_series:
            th = theta[i]
            if np.isnan(th):
                # fall back to fixed q if available, else skip
                if not use_fixed:
                    continue
                qstar = float(threshold_q)
            else:
                qstar = _ecdf_hazen(th, win)
                if np.isnan(qstar):
                    if not use_fixed:
                        continue
                    qstar = float(threshold_q)
        else:
            if not use_fixed:
                # legacy absolute-percentile mode
                conf_abs = F if tail == "high" else (1.0 - F)
                out[i] = float(np.clip(conf_abs + eps, clip_min, clip_max))
                continue
            qstar = float(threshold_q)

        # Threshold-relative mapping
        if tail == "high":
            denom = max(eps, 1.0 - qstar)
            rel = (F - qstar) / denom
        else:  # tail == "low"
            denom = max(eps, qstar)
            rel = (qstar - F) / denom

        rel = max(0.0, rel)
        out[i] = float(min(rel + eps, clip_max))

    return pd.Series(out, index=s.index)