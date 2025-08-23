# scripts/03_generate_counts.py
"""
Generate anomaly counts per evaluation window.

Compares:
  - Adaptive thresholds (per combo, from sidecar)
  - Blended families (if present)
against static IF/AE flags.

Inputs:
  - {RESULTS_RUN_DIR}/scores_with_blends.csv (preferred)
  - {RESULTS_RUN_DIR}/scores_with_thresholds.csv
  - {RESULTS_RUN_DIR}/columns_map.json

Outputs:
  - {RESULTS_RUN_DIR}/window_counts_detail.csv
  - {RESULTS_RUN_DIR}/window_counts_totals.csv
"""

from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd

import config.config as config
from src.utils import slice_by_date, write_csv, ensure_static_flags
from src.thresholding.ids import q_to_str
from src.thresholding.counting import counts_variant_vs_static


def _log(msg: str) -> None:
    print(f"[counts] {msg}")


def _selected_keys(adaptive_map: dict) -> set[str]:
    """
    Map SELECTED_COMBOS tuples (if any) to sidecar-style keys.
    """
    if not config.SELECTED_COMBOS:
        return set(adaptive_map.keys())

    want = {
        f"IFw{w_if}q{q_to_str(q_if)}_AEw{w_ae}q{q_to_str(q_ae)}"
        for (w_if, q_if, w_ae, q_ae) in config.SELECTED_COMBOS
    }
    have = set(adaptive_map.keys())
    valid = want & have
    missing = want - have
    if missing:
        _log(f"[warn] {len(missing)} SELECTED_COMBOS not in sidecar, e.g. {list(missing)[:3]}")
    if not valid:
        raise RuntimeError("After filtering, no SELECTED_COMBOS remain in sidecar.")
    return valid


def _pick_if_ae_cols_for_blend(cols: dict) -> tuple[str | None, str | None]:
    """
    Prefer dwell -> blended flags -> raw adaptive flags.
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
        raise FileNotFoundError(f"No CSV found. Tried:\n - {csv_blends}\n - {csv_base}")
    if not os.path.exists(sidecar_path):
        raise FileNotFoundError(f"Sidecar not found: {sidecar_path}")

    _log(f"Using data: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"], low_memory=False)
    with open(sidecar_path, "r", encoding="utf-8") as f:
        sidecar = json.load(f)

    # Ensure static baseline flags
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

    # Build evaluation windows
    WINDOWS = [
        {**w, "id": f"{w['label'].strip().replace(' ', '_')}_{w['start']}_{w['end']}"}
        for w in config.EVAL_WINDOWS
    ]
    if not WINDOWS:
        raise RuntimeError("No EVAL_WINDOWS defined in config.py")
    _log(f"Windows: {[w['label'] for w in WINDOWS]}")

    adaptive_map = sidecar.get("combos", {})
    blends_root  = sidecar.get("blends", {})
    scope = sidecar.get("meta", {}).get("combos_scope", "all")

    # Default valid_keys = sidecar combos
    valid_keys = set(adaptive_map.keys())

    # Legacy override (COUNT_COMBOS_SCOPE) if present
    if getattr(config, "LEGACY_COUNT_COMBOS_SCOPE", None):
        if config.LEGACY_COUNT_COMBOS_SCOPE == "selected":
            valid_keys = _selected_keys(adaptive_map)
        elif config.LEGACY_COUNT_COMBOS_SCOPE == "all":
            valid_keys = set(adaptive_map.keys())
        _log(f"[warn] Using legacy COUNT_COMBOS_SCOPE={config.LEGACY_COUNT_COMBOS_SCOPE}; "
             f"sidecar scope={scope}")

    _log(f"Counting over {len(valid_keys)} combos (sidecar scope={scope}).")

    # Run counts
    rows = []
    for w in WINDOWS:
        dfw = slice_by_date(df, "timestamp", w["start"], w["end"])

        # adaptive combos
        for cid in valid_keys:
            meta = adaptive_map[cid]
            if_col, ae_col = meta["if_flag"], meta["ae_flag"]
            if (if_col not in dfw.columns) or (ae_col not in dfw.columns):
                continue

            res = counts_variant_vs_static(dfw, if_col, ae_col,
                                           config.STATIC_IF_COL, config.STATIC_AE_COL)
            res.insert(0, "method", f"adaptive_{cid}")
            res.insert(0, "window_id", w["id"])
            res.insert(0, "window_label", w["label"])
            rows.append(res)

        # blends
        for blend_name, mapping in blends_root.items():
            for cid, cols in mapping.items():
                if cid not in valid_keys:
                    continue
                if_col, ae_col = _pick_if_ae_cols_for_blend(cols)
                if not if_col or not ae_col:
                    continue
                if (if_col not in dfw.columns) or (ae_col not in dfw.columns):
                    continue

                res = counts_variant_vs_static(dfw, if_col, ae_col,
                                               config.STATIC_IF_COL, config.STATIC_AE_COL)
                res.insert(0, "method", f"{blend_name}_{cid}")
                res.insert(0, "window_id", w["id"])
                res.insert(0, "window_label", w["label"])
                rows.append(res)

    if not rows:
        raise RuntimeError("No variants found to count.")

    detail = pd.concat(rows, ignore_index=True)

    # Aggregate totals
    totals = (detail.groupby(["window_id", "window_label", "method"], as_index=False)
              .agg(total_variant=("count_variant", "sum"),
                   total_static=("count_static", "sum"),
                   valid_n=("valid_n", "max")))
    totals["Δ_total"] = totals["total_variant"] - totals["total_static"]
    totals["rate_per_1k_variant"] = 1000 * totals["total_variant"] / totals["valid_n"].replace(0, np.nan)
    totals["rate_per_1k_static"]  = 1000 * totals["total_static"]  / totals["valid_n"].replace(0, np.nan)
    totals["Δ_rate_per_1k"]       = totals["rate_per_1k_variant"] - totals["rate_per_1k_static"]

    # Save
    detail_path = os.path.join(config.RESULTS_RUN_DIR, "window_counts_detail.csv")
    totals_path = os.path.join(config.RESULTS_RUN_DIR, "window_counts_totals.csv")
    write_csv(detail, detail_path, log_prefix="[counts]")
    write_csv(totals, totals_path, log_prefix="[counts]")

    _log(f"Saved detail -> {detail_path}")
    _log(f"Saved totals -> {totals_path}")


if __name__ == "__main__":
    main()
