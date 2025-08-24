# scripts/01_generate_thresholds.py
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", category=getattr(__import__("pandas").errors, "PerformanceWarning"))
try:
    warnings.filterwarnings("ignore", category=__import__("pandas").core.common.PerformanceWarning)
except Exception:
    pass
import os
import json
import config.config as config

from src.utils import (
    load_scores,
    write_csv,
    require_columns,
    ensure_static_flags,
)
from src.thresholding.ids import combo_id
from src.thresholding.rolling import compute_and_flag
from src.thresholding.hybrid_labelling import hybrid_from_flags
from src.utils import naming as N


def _log(msg: str) -> None:
    print(f"[thresholds] {msg}")


def main() -> None:
    # 1) Load data
    df = load_scores(config.DATA_PATH)
    require_columns(df, ["timestamp", "if_score", "lstm_score"], context="scores dataset")
    _log(f"Loaded dataset: {config.DATA_PATH}  (rows={len(df):,})")

    # 2) Work out which combos to build (scope comes from config)
    combos = list(config.combos_to_build())
    if not combos:
        raise RuntimeError(
            "No combos to build. If COMBOS_SCOPE=selected, ensure SELECTED_COMBOS is set."
        )
    _log(f"Preparing {len(combos)} combos (scope={config.combos_scope()}).")

    # 3) Sidecar scaffold
    sidecar: dict = {
        "meta": {
            "combos_scope": config.combos_scope(),
            "n_combos": len(combos),
            "adaptive_min_periods": config.ADAPTIVE_MIN_PERIODS,
        },
        "combos": {},   # filled per combo below
        "naming": {"quantile_format": "3-digit (q * 1000, zero‑padded)"},
    }

    # 4) Rolling warm‑up behavior
    #    None  -> require a full window (pandas default)
    #    1     -> start from the first row (no NA warm‑up)
    minp = None if config.ADAPTIVE_MIN_PERIODS == "window" else 1

    # 5) Compute adaptive thresholds/flags *per combo* so names carry the full combo tag
    built = 0
    for (w_if, q_if, w_ae, q_ae) in combos:
        tag = N.tag(w_if, q_if, w_ae, q_ae)           # e.g. wIF1080h_qIF030__wAE1080h_qAE975
        cid = combo_id(w_if, q_if, w_ae, q_ae)

        # IF (tail = low)
        if_thr_col, if_flag_col = compute_and_flag(
            df=df,
            score_col="if_score",
            window=w_if,
            quantile=q_if,
            model_prefix="if",
            combo_tag=tag,
            direction="low",
            min_periods=minp,
        )

        # AE (tail = high)
        ae_thr_col, ae_flag_col = compute_and_flag(
            df=df,
            score_col="lstm_score",
            window=w_ae,
            quantile=q_ae,
            model_prefix="lstm",
            combo_tag=tag,
            direction="high",
            min_periods=minp,
        )

        # Hybrid label for adaptive (no blends/dwell yet)
        hybrid_col = N.hybrid_name("adaptive", tag)
        df[hybrid_col] = hybrid_from_flags(
            df[if_flag_col].astype("boolean"),
            df[ae_flag_col].astype("boolean"),
        )

        # Record mapping for downstream steps
        sidecar["combos"][cid] = {
            "if_window": w_if,
            "if_quantile": q_if,
            "ae_window": w_ae,
            "ae_quantile": q_ae,
            "combo_tag": tag,
            "if_thresh": if_thr_col,
            "if_flag": if_flag_col,
            "ae_thresh": ae_thr_col,
            "ae_flag": ae_flag_col,
            "hybrid_label": hybrid_col,
        }
        built += 1

    _log(f"Computed thresholds, flags, and adaptive hybrid labels for {built} combos.")

    # 6) Static baseline flags (from COMP1884) — created once if missing
    created = ensure_static_flags(
        df,
        if_score_col="if_score",
        ae_score_col="lstm_score",
        static_if_thr=config.STATIC_THRESH_IF,
        static_ae_thr=config.STATIC_THRESH_AE,
        static_if_col=config.STATIC_IF_COL,
        static_ae_col=config.STATIC_AE_COL,
        force_recreate=False,
    )
    if created:
        _log(f"Created static flags: {created}")

    # 7) Save outputs
    out_csv = os.path.join(config.RESULTS_RUN_DIR, "scores_with_thresholds.csv")
    write_csv(df, out_csv, log_prefix="[thresholds]")
    _log(f"Wrote {len(df):,} rows -> {out_csv}")

    sidecar_path = os.path.join(config.RESULTS_RUN_DIR, "columns_map.json")
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)
    _log(f"Saved sidecar -> {sidecar_path}")
    _log("Done.")


if __name__ == "__main__":
    main()
