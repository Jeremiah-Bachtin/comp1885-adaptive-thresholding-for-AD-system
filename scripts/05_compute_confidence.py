# scripts/05_compute_confidence.py
from __future__ import annotations
import os, re
import pandas as pd

from config.config import (
    DATA_PATH, RESULTS_RUN_DIR, combos_to_build, resolve_min_periods,
    CONF_ENABLED, CONF_METHOD, CONF_EPS, CONF_CLIP_MIN, CONF_CLIP_MAX,
    CONF_EMIT_TYPES, CONF_COL_PREFIX,   # ← bring in the prefix
)
from src.utils import load_scores, write_csv
from src.confidence_scoring.percentile_rank import rolling_percentile_conf
from src.confidence_scoring.combine import emit_confidence_column
from src.utils import naming as N

IF_SCORE_COL = "if_score"
AE_SCORE_COL = "lstm_score"

def _log(msg: str) -> None:
    print(f"[confidence] {msg}")

def _pick_input(results_dir: str, fallback: str) -> str:
    for p in [os.path.join(results_dir, "scores_with_blends.csv"),
              os.path.join(results_dir, "scores_with_thresholds.csv")]:
        if os.path.exists(p):
            _log(f"Using input: {p}")
            return p
    _log(f"[warn] Falling back to DATA_PATH: {fallback}")
    return fallback

def _variant_and_label(df: pd.DataFrame, w_if: int, q_if: float, w_ae: int, q_ae: float) -> tuple[str, str]:
    tag = N.tag(w_if, q_if, w_ae, q_ae)
    rx_dwell = re.compile(rf"^hybrid_label_pattern_dwell\d+__{re.escape(tag)}$")
    for c in df.columns:
        if rx_dwell.fullmatch(c):
            return c.split("hybrid_label_")[1].split("__")[0], c
    rx_blend = re.compile(rf"^hybrid_label_blend_cap\d+__{re.escape(tag)}$")
    for c in df.columns:
        if rx_blend.fullmatch(c):
            return c.split("hybrid_label_")[1].split("__")[0], c
    lab = N.hybrid_name("adaptive", tag)
    if lab in df.columns:
        return "adaptive", lab
    raise RuntimeError(f"No hybrid label found for {tag}. Ensure 02_generate_blends ran (or at least 01).")

def main():
    if not CONF_ENABLED:
        _log("Skipped (CONF_ENABLED=0).")
        return
    if CONF_METHOD != "percentile_rank":
        raise NotImplementedError(f"Unsupported method: {CONF_METHOD}")

    in_csv = _pick_input(RESULTS_RUN_DIR, DATA_PATH)
    df = load_scores(in_csv)

    created_cols = []
    for (w_if_h, q_if, w_ae_h, q_ae) in combos_to_build():
        tag = N.tag(w_if_h, q_if, w_ae_h, q_ae)

        mp_if = resolve_min_periods(w_if_h)
        mp_ae = resolve_min_periods(w_ae_h)

        conf_if_col = f"{CONF_COL_PREFIX}_if__{tag}"
        conf_ae_col = f"{CONF_COL_PREFIX}_ae__{tag}"

        # Threshold‑relative percentile ranks (near-threshold ≈ 0, deep-tail → 1)
        df[conf_if_col] = rolling_percentile_conf(
            df[IF_SCORE_COL], tail="low", window_hours=w_if_h, min_periods=mp_if,
            eps=CONF_EPS, clip_min=0.0, clip_max=CONF_CLIP_MAX, threshold_q=q_if
        )
        df[conf_ae_col] = rolling_percentile_conf(
            df[AE_SCORE_COL], tail="high", window_hours=w_ae_h, min_periods=mp_ae,
            eps=CONF_EPS, clip_min=0.0, clip_max=CONF_CLIP_MAX, threshold_q=q_ae
        )

        variant, label_col = _variant_and_label(df, w_if_h, q_if, w_ae_h, q_ae)
        conf_out_col = N.conf_final(tag, variant)  # e.g. conf__pattern_dwell3__wIF...__wAE...

        df = emit_confidence_column(
            df,
            label_col=label_col,
            if_conf_col=conf_if_col,
            ae_conf_col=conf_ae_col,
            out_col=conf_out_col,
            emit_types=set(CONF_EMIT_TYPES),
        )

        created_cols.extend([conf_if_col, conf_ae_col, conf_out_col])

    out_csv = os.path.join(RESULTS_RUN_DIR, "complete_scores_thresholds_confidence.csv")
    write_csv(df, out_csv, log_prefix="[confidence]")
    _log(f"Wrote {len(created_cols)} columns → {out_csv}")

if __name__ == "__main__":
    main()
