# scripts/05_compute_confidence.py
from __future__ import annotations
import os
import re
import numpy as np
import pandas as pd

from config.config import (
    DATA_PATH, RESULTS_RUN_DIR, resolve_min_periods,
    CONF_ENABLED, CONF_METHOD, CONF_EPS, CONF_CLIP_MIN, CONF_CLIP_MAX,
    CONF_EMIT_TYPES, CONF_COL_PREFIX,
    CONF_COMPOUND_AGG, CONF_POINT_SOURCE, CONF_PATTERN_SOURCE,
)
from src.utils import load_scores, write_csv
from src.confidence_scoring.percentile_rank import rolling_percentile_conf
from src.utils import naming as N

IF_SCORE_COL = "if_score"
AE_SCORE_COL = "lstm_score"

def _log(msg: str) -> None:
    print(f"[confidence] {msg}", flush=True)

def _pick_input(results_dir: str, fallback: str) -> str:
    for p in [os.path.join(results_dir, "scores_with_blends.csv"),
              os.path.join(results_dir, "scores_with_thresholds.csv")]:
        if os.path.exists(p):
            _log(f"Using input: {p}")
            return p
    _log(f"[warn] Falling back to DATA_PATH: {fallback}")
    return fallback

# --- parse helpers -----------------------------------------------------------

_RX_HYB = re.compile(r"^hybrid_label_(?P<variant>[^_][^_]*(?:_[^_]+)*)__(?P<tag>wIF\d+h_qIF\d{3}__wAE\d+h_qAE\d{3})$")
_RX_TAG = re.compile(r"^wIF(?P<wif>\d+)h_qIF(?P<qif>\d{3})__wAE(?P<wae>\d+)h_qAE(?P<qae>\d{3})$")

def _parse_hybrid_col(col: str) -> tuple[str, str]:
    """
    Return (variant, tag) from a hybrid_label column name.
    Expects: hybrid_label_{variant}__{tag}
    """
    m = _RX_HYB.match(col)
    if not m:
        raise ValueError(f"Bad hybrid column name: {col}")
    return m.group("variant"), m.group("tag")

def _parse_tag(tag: str) -> tuple[int, float, int, float]:
    """
    Convert combo tag -> (w_if_h, q_if, w_ae_h, q_ae)
    """
    m = _RX_TAG.match(tag)
    if not m:
        raise ValueError(f"Bad combo tag: {tag}")
    w_if_h = int(m.group("wif"))
    q_if   = int(m.group("qif")) / 1000.0
    w_ae_h = int(m.group("wae"))
    q_ae   = int(m.group("qae")) / 1000.0
    return w_if_h, q_if, w_ae_h, q_ae

# --- combine logic (inlined to avoid previous NA/signature issues) -----------

def _combine_confidence(
    label: pd.Series,
    conf_if: pd.Series,
    conf_ae: pd.Series,
    *,
    emit_types: set[str],
    compound_agg: str,
    point_source: str,
    pattern_source: str,
) -> pd.Series:
    """
    Build one final confidence series based on the anomaly label and policy knobs.
    """
    out = pd.Series(np.nan, index=label.index, dtype="float64")

    # Normalize label values to Title case ('Point','Pattern','Compound','None', …)
    lbl = label.fillna("None").astype(str).str.title()

    # Where to emit
    emit_mask = lbl.isin({t.title() for t in emit_types})
    if not emit_mask.any():
        return out  # nothing to emit

    # Compound aggregation
    if compound_agg == "mean":
        compound_vals = (conf_if + conf_ae) / 2.0
    elif compound_agg == "min":
        compound_vals = np.minimum(conf_if, conf_ae)
    elif compound_agg == "max":
        compound_vals = np.maximum(conf_if, conf_ae)
    else:
        # fallback to mean if unknown (keeps it robust)
        compound_vals = (conf_if + conf_ae) / 2.0

    # Point source
    point_vals = conf_if if point_source.upper() == "IF" else conf_ae
    # Pattern source
    pattern_vals = conf_ae if pattern_source.upper() == "AE" else conf_if

    # Assign per class (respect emit mask)
    m_point    = emit_mask & (lbl == "Point")
    m_pattern  = emit_mask & (lbl == "Pattern")
    m_compound = emit_mask & (lbl == "Compound")

    out.loc[m_point]    = point_vals.loc[m_point].astype(float)
    out.loc[m_pattern]  = pattern_vals.loc[m_pattern].astype(float)
    out.loc[m_compound] = compound_vals.loc[m_compound].astype(float)

    return out

# --- main --------------------------------------------------------------------

def main():
    if not CONF_ENABLED:
        _log("Skipped (CONF_ENABLED=0).")
        return
    if CONF_METHOD != "percentile_rank":
        raise NotImplementedError(f"Unsupported method: {CONF_METHOD}")

    in_csv = _pick_input(RESULTS_RUN_DIR, DATA_PATH)
    df = load_scores(in_csv)

    # Find every hybrid label column present
    hybrid_cols = [c for c in df.columns if c.startswith("hybrid_label_")]
    if not hybrid_cols:
        raise RuntimeError("No hybrid_label_* columns found. Run 01 & 02 first.")

    created = []
    for lab_col in hybrid_cols:
        try:
            variant, tag = _parse_hybrid_col(lab_col)
            w_if_h, q_if, w_ae_h, q_ae = _parse_tag(tag)
        except Exception as e:
            _log(f"[warn] Skipping {lab_col}: {e}")
            continue

        # rolling windows + warm-up aligned to thresholds
        mp_if = resolve_min_periods(w_if_h)
        mp_ae = resolve_min_periods(w_ae_h)

        # Per‑model confidences (threshold‑relative mode enabled via threshold_q)
        conf_if = rolling_percentile_conf(
            df[IF_SCORE_COL], tail="low", window_hours=w_if_h, min_periods=mp_if,
            eps=CONF_EPS, clip_min=CONF_CLIP_MIN, clip_max=CONF_CLIP_MAX,
            threshold_q=q_if
        )
        conf_ae = rolling_percentile_conf(
            df[AE_SCORE_COL], tail="high", window_hours=w_ae_h, min_periods=mp_ae,
            eps=CONF_EPS, clip_min=CONF_CLIP_MIN, clip_max=CONF_CLIP_MAX,
            threshold_q=q_ae
        )

        # Final column name: conf__{variant}__{tag}
        out_col = N.conf_final(tag, variant)

        # Build combined confidence according to policy and write it
        df[out_col] = _combine_confidence(
            df[lab_col], conf_if, conf_ae,
            emit_types=set(CONF_EMIT_TYPES),
            compound_agg=CONF_COMPOUND_AGG,
            point_source=CONF_POINT_SOURCE,
            pattern_source=CONF_PATTERN_SOURCE,
        )
        created.append(out_col)

    out_csv = os.path.join(RESULTS_RUN_DIR, "complete_scores_thresholds_confidence.csv")
    write_csv(df, out_csv, log_prefix="[confidence]")
    _log(f"Appended {len(created)} confidence columns → {out_csv}")

if __name__ == "__main__":
    main()
