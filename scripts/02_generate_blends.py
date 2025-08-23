# scripts/02_generate_blends.py
"""
Build per-combo blended variants (capped min–max, dwell) from adaptive outputs.

Inputs (must exist from 01_generate_thresholds.py):
  - {RESULTS_RUN_DIR}/scores_with_thresholds.csv
  - {RESULTS_RUN_DIR}/columns_map.json

Reads .env via config.BLEND_SPECS, e.g.:
  BLENDS=CAPPED_minmax,DWELL_PATTERN
  BLEND_CAPPED_minmax=use:all;op:capped_minmax;cap:0.02
  BLEND_DWELL_PATTERN=use:selected;op:dwell_pattern;k:3

Outputs:
  - {RESULTS_RUN_DIR}/scores_with_blends.csv   (same data + new blend columns)
  - {RESULTS_RUN_DIR}/columns_map.json         (updated with per-combo blend mapping)
"""

from __future__ import annotations
import os
import json
import pandas as pd

import config.config as config
from src.thresholding.ids import combo_id
from src.thresholding.blending import (
    capped_minmax_thresholds,
    flags_from_thresholds,
    dwell_on_pattern_only,
)
from src.thresholding.hybrid_labelling import hybrid_from_flags
from src.utils import write_csv


def _log(msg: str) -> None:
    print(f"[blend] {msg}")


def _fmt_window(h: int) -> str:
    """Format hours as 'Xd (Yh)' for logging clarity."""
    d = h / 24
    return f"{d:.1f}d ({h}h)"


def _selected_combo_ids(all_combos: dict) -> list[str]:
    """
    Return list of combo_ids to operate on (respect SELECTED_COMBOS when present).
    """
    if not config.SELECTED_COMBOS:
        return list(all_combos.keys())

    keys: list[str] = []
    for (w_if, q_if, w_ae, q_ae) in config.SELECTED_COMBOS:
        cid = combo_id(w_if, q_if, w_ae, q_ae)
        if cid in all_combos:
            keys.append(cid)
    if not keys:
        raise RuntimeError("After filtering, no SELECTED_COMBOS remain in sidecar.")
    return keys


def _resolve_best_flag_cols(sidecar: dict, cid: str) -> tuple[str | None, str | None]:
    """
    Choose the best available IF/AE flag columns for dwell:
      prefer most recent blended flags if present, else adaptive flags.
    """
    blends_root = sidecar.get("blends", {})
    if blends_root:
        for name in reversed(list(blends_root.keys())):
            mapping = blends_root[name] or {}
            if cid in mapping:
                if_flag = mapping[cid].get("if_blend_flag") or mapping[cid].get("if_flag")
                ae_flag = mapping[cid].get("ae_blend_flag") or mapping[cid].get("ae_flag")
                if if_flag and ae_flag:
                    return if_flag, ae_flag

    meta = sidecar["combos"].get(cid, {})
    return meta.get("if_flag"), meta.get("ae_flag")


def main() -> None:
    # Inputs produced by 01_generate_thresholds.py
    csv_in = os.path.join(config.RESULTS_RUN_DIR, "scores_with_thresholds.csv")
    sidecar_path = os.path.join(config.RESULTS_RUN_DIR, "columns_map.json")

    if not os.path.exists(csv_in) or not os.path.exists(sidecar_path):
        raise FileNotFoundError(
            "Missing inputs. Run scripts/01_generate_thresholds.py first.\n"
            f"CSV: {csv_in}\nSidecar: {sidecar_path}"
        )

    df = pd.read_csv(csv_in, parse_dates=["timestamp"], low_memory=False)
    with open(sidecar_path, "r", encoding="utf-8") as f:
        sidecar = json.load(f)

    all_combos: dict = sidecar.get("combos", {})
    if not all_combos:
        _log("No combos found in sidecar; nothing to do.")
        return

    if not config.BLEND_SPECS:
        _log("No BLEND_SPECS configured; nothing to do.")
        # still copy thresholds.csv to blends.csv for downstream convenience
        csv_out = os.path.join(config.RESULTS_RUN_DIR, "scores_with_blends.csv")
        write_csv(df, csv_out, log_prefix="[blend]")
        return

    # Clarified logging: show total combos and how many are in SELECTED_COMBOS
    total = len(all_combos)
    selected = _selected_combo_ids(all_combos)
    scope = sidecar.get("meta", {}).get("combos_scope", "all")
    _log(f"Found {total} combos in sidecar (scope={scope}); {len(selected)} in SELECTED_COMBOS.")

    for (w_if, q_if, w_ae, q_ae) in config.SELECTED_COMBOS or []:
        _log(f"  Combo: IF={_fmt_window(w_if)} @ q={q_if}, "
             f"AE={_fmt_window(w_ae)} @ q={q_ae}")

    # Ensure sidecar has a place for blends
    sidecar.setdefault("blends", {})

    # Apply each configured blend spec in the order declared in .env
    for name, spec in config.BLEND_SPECS.items():
        op = (spec.get("op") or "").strip()
        use = (spec.get("use") or "selected").strip()
        cids = list(all_combos.keys()) if use == "all" else selected
        _log(f"Applying '{name}' (op={op}, use={use}, combos={len(cids)}/{total})")

        if op == "capped_minmax":
            cap = float(spec.get("cap", 0.1))
            sidecar["blends"].setdefault(name, {})

            for cid in cids:
                meta = all_combos[cid]
                if_thr = meta.get("if_thresh")
                ae_thr = meta.get("ae_thresh")
                if not if_thr or not ae_thr:
                    raise KeyError(f"Combo {cid} missing adaptive threshold columns in sidecar.")

                # 1) Blend thresholds
                thr_cols = capped_minmax_thresholds(
                    df,
                    if_adaptive_thr_col=if_thr,
                    ae_adaptive_thr_col=ae_thr,
                    static_if_thr=config.STATIC_THRESH_IF,
                    static_ae_thr=config.STATIC_THRESH_AE,
                    cap_delta=cap,
                )

                # 2) Flags from blended thresholds
                flag_cols = flags_from_thresholds(
                    df,
                    if_score_col="if_score",
                    ae_score_col="lstm_score",
                    if_thr_col=thr_cols["if_blend_thresh"],
                    ae_thr_col=thr_cols["ae_blend_thresh"],
                )

                # 3) Hybrid label from centralised helper
                hybrid_col = f"hybrid_label_blend_cap{int(cap*1000):03d}__{cid}"
                df[hybrid_col] = hybrid_from_flags(
                    df[flag_cols["if_flag_col"]].astype("boolean"),
                    df[flag_cols["ae_flag_col"]].astype("boolean"),
                )

                # 4) Update sidecar mapping
                sidecar["blends"][name][cid] = {
                    "if_blend_thresh": thr_cols["if_blend_thresh"],
                    "ae_blend_thresh": thr_cols["ae_blend_thresh"],
                    "if_blend_flag": flag_cols["if_flag_col"],
                    "ae_blend_flag": flag_cols["ae_flag_col"],
                    "hybrid_label": hybrid_col,
                    "if_flag": meta.get("if_flag"),
                    "ae_flag": meta.get("ae_flag"),
                }

        elif op == "dwell_pattern":
            k = int(spec.get("k", 3))
            sidecar["blends"].setdefault(name, {})

            for cid in cids:
                if_flag, ae_flag = _resolve_best_flag_cols(sidecar, cid)
                if not if_flag or not ae_flag:
                    _log(f"[warn] Missing flags for {cid}; skipping dwell_pattern.")
                    continue

                dwell_cols = dwell_on_pattern_only(df, if_flag, ae_flag, k=k)

                # Hybrid label (centralised helper)
                hybrid_col = dwell_cols["hybrid_col"]
                df[hybrid_col] = hybrid_from_flags(
                    df[dwell_cols["if_flag_col"]].astype("boolean"),
                    df[dwell_cols["ae_dwell_col"]].astype("boolean"),
                )

                sidecar["blends"][name][cid] = {
                    "if_flag": dwell_cols["if_flag_col"],  # unchanged IF
                    "ae_flag_dwell": dwell_cols["ae_dwell_col"],
                    "hybrid_label": hybrid_col,
                }

        else:
            raise ValueError(f"Unsupported blend op: {op}")

    # ---- Save updated CSV + sidecar ---------------------------------------
    csv_out = os.path.join(config.RESULTS_RUN_DIR, "scores_with_blends.csv")
    write_csv(df, csv_out, log_prefix="[blend]")

    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)
    _log(f"Updated sidecar -> {sidecar_path}")


if __name__ == "__main__":
    main()
