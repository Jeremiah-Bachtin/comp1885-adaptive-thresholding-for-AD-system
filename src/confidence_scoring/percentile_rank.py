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
    # NEW: pass the combo's threshold quantile to enable threshold‑relative normalization
    threshold_q: Optional[float] = None,
) -> pd.Series:
    """
    Percentile‑rank confidence over a trailing window that EXCLUDES the current row.

    Modes
    -----
    1) Absolute percentile (legacy, default when threshold_q is None)
       - AE (tail='high'): conf =  ECDF(x_t)
       - IF (tail='low') : conf = 1 - ECDF(x_t)

    2) Threshold‑relative percentile (when threshold_q is provided)
       Let F = ECDF(x_t) in [0,1], and q = threshold_q for the model.

       - AE (tail='high', anomaly when F >= q):
           conf = max(0, (F - q) / max(eps, 1 - q))
       - IF (tail='low', anomaly when F <= q):
           conf = max(0, (q - F) / max(eps, q))

       This yields:
         • ~0 when just at/over the threshold (low confidence),
         • →1 as the point moves deeper into the anomalous tail (high confidence),
         • 0 for non‑anomalous points (you can still mask by label upstream/downstream).

    Notes
    -----
    - Uses a trailing window of `window_hours` rows (pipeline treats rows as hourly).
    - `min_periods` controls warm‑up exactly as your thresholds do.
    - Final values are clipped to [clip_min, clip_max] after adding eps, to avoid exact 0/1.
    """
    s = scores.astype(float)
    s_prev = s.shift(1)  # exclude current to avoid look‑ahead
    out = np.full(len(s), np.nan, dtype=float)
    vals, prev = s.to_numpy(), s_prev.to_numpy()

    use_relative = threshold_q is not None
    q = float(threshold_q) if use_relative else None

    for i in range(len(vals)):
        start = max(0, i - window_hours)
        win = prev[start:i]
        if (i - start) < min_periods:
            continue

        F = _ecdf_hazen(vals[i], win)
        if np.isnan(F):
            continue

        if not use_relative:
            # Legacy absolute percentile mode
            conf_abs = F if tail == "high" else (1.0 - F)
            out[i] = float(np.clip(conf_abs + eps, clip_min, clip_max))
        else:
            # Threshold‑relative mode
            if tail == "high":
                # anomaly when F >= q
                denom = max(eps, 1.0 - q)
                rel = (F - q) / denom
            else:
                # tail == "low"; anomaly when F <= q
                denom = max(eps, q)
                rel = (q - F) / denom

            rel = max(0.0, rel)  # 0 if not past the threshold
            out[i] = float(min(rel + eps, clip_max))

    return pd.Series(out, index=s.index)
