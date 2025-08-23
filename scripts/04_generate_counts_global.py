# scripts/04_generate_counts_global.py
"""
Global summary counts.

For every combo in sidecar (and every present blend), compute total
Point/Pattern/Compound counts over the entire dataset, using only rows where
that variant's flags exist, and compare to static on the same mask.

Inputs (prefer blended dataset if present):
  - {RESULTS_RUN_DIR}/scores_with_blends.csv    OR
  - {RESULTS_RUN_DIR}/scores_with_thresholds.csv
  - {RESULTS_RUN_DIR}/columns_map.json

Output:
  - {RESULTS_RUN_DIR}/global_counts_summary.csv
"""

from __future__ import annotations
import os
import json
import pandas as pd

import config.config as config
from src.thresholding.counting import counts_variant_vs_static
from src.utils import write_csv, ensure_static_flags


def _log(msg: str) -> None:
    print(f"[counts:global] {msg}")


def _pick_if_ae_cols_for_blend(cols: dict) -> tuple[str | None, str | None]:
    """
    Given sidecar blend mapping for a combo, pick the best IF/AE flag cols:
    prefer dwell -> blended flags -> raw adaptive flags.
    """
    if_col = cols.get("if_flag_dwell") or cols.get("if_blend_flag") or cols.get("if_flag")
    ae_col = cols.get("ae_flag_dwell") or cols.get("ae_blend_flag") or cols.get("ae_flag")
    return if_col, ae_col


def main() -> None:
    # Prefer blended dataset if available
    csv_blends = os.path.join(config.RESULTS_RUN_DIR, "scores_with_blends.csv")
    csv_base   = os.path.join(config.RESULTS_RUN_DIR, "scores_with_thresholds.csv")
    sidecar_path = os.path.join(config.RESULTS_RUN_DIR, "columns_map.json")

    csv_path = csv_blends if os.path.exists(csv_blends) else csv_base
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"No input CSV found.\nTried:\n - {csv_blends}\n - {csv_base}"
        )
    if not os.path.exists(sidecar_path):
        raise FileNotFoundError(f"Sidecar not found: {sidecar_path}")

    _log(f"Using data: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"], low_memory=False)
    with open(sidecar_path, "r", encoding="utf-8") as f:
        sidecar = json.load(f)

    # Ensure static baseline flags exist
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

    adaptive_map = sidecar.get("combos", {})  # combo_id -> {if_flag, ae_flag, ...}
    blends_root  = sidecar.get("blends", {})  # blend_name -> {combo_id -> {...}}
    scope = sidecar.get("meta", {}).get("combos_scope", "all")

    # Default valid_keys = sidecar combos
    valid_keys = set(adaptive_map.keys())

    # Legacy override if present
    if getattr(config, "LEGACY_COUNT_COMBOS_SCOPE", None):
        if config.LEGACY_COUNT_COMBOS_SCOPE == "selected":
            from src.thresholding.ids import combo_id, q_to_str
            want = {
                f"IFw{w_if}q{q_to_str(q_if)}_AEw{w_ae}q{q_to_str(q_ae)}"
                for (w_if, q_if, w_ae, q_ae) in (config.selected_combo_tuples() or [])
            }
            valid_keys = want & set(adaptive_map.keys())
        elif config.LEGACY_COUNT_COMBOS_SCOPE == "all":
            valid_keys = set(adaptive_map.keys())
        _log(f"[warn] Using legacy COUNT_COMBOS_SCOPE={config.LEGACY_COUNT_COMBOS_SCOPE}; "
             f"sidecar scope={scope}")

    _log(f"Counting global totals over {len(valid_keys)} combos (scope={scope}).")

    rows = []

    # --- Adaptive combos (no blending) ---
    for cid, meta in adaptive_map.items():
        if cid not in valid_keys:
            continue
        if_col = meta.get("if_flag")
        ae_col = meta.get("ae_flag")
        if not if_col or not ae_col:
            continue
        if (if_col not in df.columns) or (ae_col not in df.columns):
            continue

        res = counts_variant_vs_static(df, if_col, ae_col,
                                       config.STATIC_IF_COL, config.STATIC_AE_COL)
        res.insert(0, "method", f"adaptive_{cid}")
        rows.append(res)

    # --- Blended families (if present) ---
    for blend_name, mapping in blends_root.items():
        for cid, cols in mapping.items():
            if cid not in valid_keys:
                continue
            if_col, ae_col = _pick_if_ae_cols_for_blend(cols)
            if not if_col or not ae_col:
                continue
            if (if_col not in df.columns) or (ae_col not in df.columns):
                continue

            res = counts_variant_vs_static(df, if_col, ae_col,
                                           config.STATIC_IF_COL, config.STATIC_AE_COL)
            res.insert(0, "method", f"{blend_name}_{cid}")
            rows.append(res)

    if rows:
        summary = pd.concat(rows, ignore_index=True)
    else:
        raise RuntimeError("No variants found to count globally.")

    # Compute diffs and percentages
    summary["diff_abs"] = summary["count_variant"] - summary["count_static"]
    denom = summary["count_static"].replace(0, pd.NA)
    pct = (summary["diff_abs"] / denom * 100).round()
    summary["diff_pct"] = pct.astype("Int64").astype(str) + "%"

    # Ensure ordering of anomaly types
    summary["anomaly_type"] = pd.Categorical(
        summary["anomaly_type"],
        ["Point", "Pattern", "Compound"],
        ordered=True,
    )
    summary = summary.sort_values(["method", "anomaly_type"]).reset_index(drop=True)

    # Rename to match adaptive/static wording
    summary = summary.rename(columns={
        "count_variant": "count_adaptive",
        "count_static": "count_static",
    })

    # Final column order
    summary = summary[[
        "method", "anomaly_type",
        "count_adaptive", "count_static",
        "diff_abs", "diff_pct", "valid_n"
    ]]

    out_path = os.path.join(config.RESULTS_RUN_DIR, "global_counts_summary.csv")
    write_csv(summary, out_path, log_prefix="[counts:global]")
    _log(f"Saved summary -> {out_path}")


if __name__ == "__main__":
    main()
