# scripts/03_generate_counts.py
from __future__ import annotations
import os
import json
import pandas as pd
from typing import Tuple

import config.config as config
from src.utils import load_scores, write_csv, ensure_static_flags
from src.thresholding.counting import counts_variant_vs_static

IF_SCORE_COL = "if_score"
AE_SCORE_COL = "lstm_score"
TS_COL = "timestamp"


def _log(msg: str) -> None:
    print(f"[counts] {msg}")


def _pick_input(results_dir: str, fallback: str) -> str:
    for p in [os.path.join(results_dir, "scores_with_blends.csv"),
              os.path.join(results_dir, "scores_with_thresholds.csv")]:
        if os.path.exists(p):
            _log(f"Using data: {p}")
            return p
    _log(f"[warn] Falling back to DATA_PATH: {fallback}")
    return fallback


def _load_sidecar(results_dir: str) -> dict:
    sidecar_path = os.path.join(results_dir, "columns_map.json")
    if not os.path.exists(sidecar_path):
        raise FileNotFoundError("columns_map.json not found. Run 01_generate_thresholds.py first.")
    with open(sidecar_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_windows() -> list[dict]:
    """config.EVAL_WINDOWS is a list of dicts: {label,start,end}."""
    win_dicts = config.EVAL_WINDOWS or []
    out = []
    for w in win_dicts:
        label = w.get("label")
        start = pd.to_datetime(w.get("start"))
        end = pd.to_datetime(w.get("end"))
        if label is None or pd.isna(start) or pd.isna(end):
            raise ValueError(f"Bad window spec: {w}")
        out.append({"label": label, "start": start, "end": end})
    return out


def _window_id(w: dict) -> str:
    return f"{w['label']}_{w['start'].strftime('%Y-%m-%d')}_{w['end'].strftime('%Y-%m-%d')}"


def _pick_if_ae_cols_for_blend(cols: dict) -> Tuple[str | None, str | None]:
    """Prefer dwell -> blended flags -> raw adaptive flags."""
    if_col = cols.get("if_flag_dwell") or cols.get("if_blend_flag") or cols.get("if_flag")
    ae_col = cols.get("ae_flag_dwell") or cols.get("ae_blend_flag") or cols.get("ae_flag")
    return if_col, ae_col


def _emit_rows_for_variant(dfw: pd.DataFrame, win: dict, method_name: str,
                           if_col: str, ae_col: str) -> pd.DataFrame:
    res = counts_variant_vs_static(dfw, if_col, ae_col,
                                   config.STATIC_IF_COL, config.STATIC_AE_COL)
    # Decorate with window & method
    res.insert(0, "method", method_name)
    res.insert(0, "window_id", _window_id(win))
    res.insert(1, "window_label", win["label"])
    return res


def main() -> None:
    # Load data
    csv_in = _pick_input(config.RESULTS_RUN_DIR, config.DATA_PATH)
    df = load_scores(csv_in)
    if TS_COL not in df.columns:
        raise KeyError(f"Missing timestamp column '{TS_COL}' in dataset.")
    df[TS_COL] = pd.to_datetime(df[TS_COL])

    # Ensure static baseline flags exist
    ensure_static_flags(
        df,
        if_score_col=IF_SCORE_COL,
        ae_score_col=AE_SCORE_COL,
        static_if_thr=config.STATIC_THRESH_IF,
        static_ae_thr=config.STATIC_THRESH_AE,
        static_if_col=config.STATIC_IF_COL,
        static_ae_col=config.STATIC_AE_COL,
        force_recreate=False,
    )

    # Windows
    windows = _parse_windows()
    if not windows:
        _log("No EVAL_WINDOWS configured; nothing to count.")
        return
    _log(f"Windows: {[w['label'] for w in windows]}")

    # Sidecar (combos + blends)
    sidecar = _load_sidecar(config.RESULTS_RUN_DIR)
    combos = sidecar.get("combos", {})
    blends_root = sidecar.get("blends", {})
    if not combos:
        _log("No combos found in sidecar; nothing to count.")
        return
    scope = sidecar.get("meta", {}).get("combos_scope", "all")
    _log(f"Counting over {len(combos)} combos (sidecar scope={scope}).")

    detail_rows: list[pd.DataFrame] = []

    for w in windows:
        # inclusive slice
        m = (df[TS_COL] >= w["start"]) & (df[TS_COL] <= w["end"])
        dfw = df.loc[m].copy()

        # Adaptive per-combo
        for cid, meta in combos.items():
            if_col = meta.get("if_flag")
            ae_col = meta.get("ae_flag")
            if not if_col or not ae_col:
                continue
            if (if_col not in dfw.columns) or (ae_col not in dfw.columns):
                continue
            method = f"adaptive_{cid}"
            detail_rows.append(_emit_rows_for_variant(dfw, w, method, if_col, ae_col))

        # Each blend family
        for blend_name, mapping in blends_root.items():
            for cid, cols in mapping.items():
                if_col, ae_col = _pick_if_ae_cols_for_blend(cols)
                if not if_col or not ae_col:
                    continue
                if (if_col not in dfw.columns) or (ae_col not in dfw.columns):
                    continue
                method = f"{blend_name}_{cid}"
                detail_rows.append(_emit_rows_for_variant(dfw, w, method, if_col, ae_col))

    if not detail_rows:
        _log("No rows created; check sidecar/columns.")
        return

    detail = pd.concat(detail_rows, ignore_index=True)

    # Totals per window & method
    totals = (
        detail.groupby(["window_id", "window_label", "method"], as_index=False)
              .agg(total_variant=("count_variant", "sum"),
                   total_static=("count_static", "sum"),
                   valid_n=("valid_n", "min"))
    )
    totals["Δ_total"] = totals["total_variant"] - totals["total_static"]
    denom = totals["valid_n"].replace(0, pd.NA)
    totals["rate_per_1k_variant"] = (totals["total_variant"] / denom * 1000).round(6)
    totals["rate_per_1k_static"]  = (totals["total_static"] / denom * 1000).round(6)
    totals["Δ_rate_per_1k"] = (totals["rate_per_1k_variant"] - totals["rate_per_1k_static"]).round(6)

    # Sort for readability
    detail = detail.sort_values(["window_id", "method", "anomaly_type"]).reset_index(drop=True)
    totals = totals.sort_values(["window_id", "method"]).reset_index(drop=True)

    # Write outputs (same filenames as legacy)
    out_detail = os.path.join(config.RESULTS_RUN_DIR, "window_counts_detail.csv")
    out_totals = os.path.join(config.RESULTS_RUN_DIR, "window_counts_totals.csv")
    write_csv(detail, out_detail, log_prefix="[counts]")
    write_csv(totals, out_totals, log_prefix="[counts]")
    _log(f"Saved detail -> {out_detail}")
    _log(f"Saved totals -> {out_totals}")


if __name__ == "__main__":
    main()
