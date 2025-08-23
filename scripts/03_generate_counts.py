# scripts/03_generate_counts.py
from __future__ import annotations
import os
import json
import pandas as pd

import config.config as config
from src.utils import load_scores, write_csv

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

def _hybrid_col_for_combo(df: pd.DataFrame, sidecar: dict, cid: str) -> str:
    """
    Prefer latest variant: dwell -> blend -> adaptive.
    Sidecar may or may not have blends; fall back robustly.
    """
    # dwell (if present)
    for blend_name in ["DWELL_PATTERN", "Dwell_Pattern", "dwell_pattern"]:
        col = sidecar.get("blends", {}).get(blend_name, {}).get(cid, {}).get("hybrid_label")
        if col and col in df.columns:
            return col
    # capped minmax
    for blend_name in ["CAPPED_minmax", "Capped_Minmax", "capped_minmax"]:
        col = sidecar.get("blends", {}).get(blend_name, {}).get(cid, {}).get("hybrid_label")
        if col and col in df.columns:
            return col
    # adaptive (always written by step 01)
    col = sidecar.get("combos", {}).get(cid, {}).get("hybrid_label")
    if col and col in df.columns:
        return col
    raise RuntimeError(f"No hybrid label column found in dataframe for combo id {cid}.")

def _parse_windows() -> list[dict]:
    """
    config.EVAL_WINDOWS is a list of dicts: {label,start,end}.
    Normalize to Timestamp bounds and validate.
    """
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

def _count_types(series: pd.Series) -> dict:
    """
    Count anomaly label categories. Expected values: 'Point','Pattern','Compound','None' (or NaN).
    """
    vc = series.value_counts(dropna=False)
    get = lambda k: int(vc.get(k, 0))
    none_ct = int(vc.get("None", 0)) + int(vc.get(None, 0))  # be tolerant
    return {
        "n_point": get("Point"),
        "n_pattern": get("Pattern"),
        "n_compound": get("Compound"),
        "n_none": none_ct
    }

def main() -> None:
    # Load data
    csv_in = _pick_input(config.RESULTS_RUN_DIR, config.DATA_PATH)
    df = load_scores(csv_in)
    if TS_COL not in df.columns:
        raise KeyError(f"Missing timestamp column '{TS_COL}' in dataset.")
    df[TS_COL] = pd.to_datetime(df[TS_COL])

    # Windows (dict-based)
    windows = _parse_windows()
    if not windows:
        _log("No EVAL_WINDOWS configured; nothing to count.")
        return
    _log(f"Windows: {[w['label'] for w in windows]}")

    # Sidecar for combos and (if present) blend outputs
    sidecar = _load_sidecar(config.RESULTS_RUN_DIR)
    combos = sidecar.get("combos", {})
    if not combos:
        _log("No combos found in sidecar; nothing to count.")
        return
    scope = sidecar.get("meta", {}).get("combos_scope", "all")
    _log(f"Counting over {len(combos)} combos (sidecar scope={scope}).")

    # Build detail rows
    detail_rows = []
    for cid, meta in combos.items():
        # pick hybrid label column (latest available)
        hybrid_col = _hybrid_col_for_combo(df, sidecar, cid)

        for w in windows:
            label = w["label"]
            start, end = w["start"], w["end"]
            mask = (df[TS_COL] >= start) & (df[TS_COL] <= end)
            window_slice = df.loc[mask]

            counts = _count_types(window_slice[hybrid_col])

            detail_rows.append({
                "window": label,
                "start": start.strftime("%Y-%m-%d"),
                "end": end.strftime("%Y-%m-%d"),
                "combo_id": cid,
                "if_window_h": meta.get("if_window"),
                "if_q": meta.get("if_quantile"),
                "ae_window_h": meta.get("ae_window"),
                "ae_q": meta.get("ae_quantile"),
                "hybrid_col": hybrid_col,
                "n_rows": int(window_slice.shape[0]),
                **counts,
                "n_anom": counts["n_point"] + counts["n_pattern"] + counts["n_compound"],
            })

    detail_df = pd.DataFrame(detail_rows)

    # Totals per window (sum over combos)
    totals = (
        detail_df
        .groupby(["window", "start", "end"], as_index=False)[
            ["n_rows", "n_point", "n_pattern", "n_compound", "n_none", "n_anom"]
        ]
        .sum()
        .sort_values(["window"])
        .reset_index(drop=True)
    )

    # Write outputs
    out_detail = os.path.join(config.RESULTS_RUN_DIR, "window_counts_detail.csv")
    out_totals = os.path.join(config.RESULTS_RUN_DIR, "window_counts_totals.csv")
    write_csv(detail_df, out_detail, log_prefix="[counts]")
    write_csv(totals, out_totals, log_prefix="[counts]")

    _log(f"Saved detail -> {out_detail}")
    _log(f"Saved totals -> {out_totals}")

if __name__ == "__main__":
    main()
