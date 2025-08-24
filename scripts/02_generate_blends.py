# scripts/02_generate_blends.py
from __future__ import annotations
import os
import json
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=getattr(__import__("pandas").errors, "PerformanceWarning"))
try:
    warnings.filterwarnings("ignore", category=__import__("pandas").core.common.PerformanceWarning)
except Exception:
    pass
import config.config as config
from src.utils import write_csv
from src.utils import naming as N
from src.thresholding.ids import combo_id
from src.thresholding.blending import (
    capped_minmax_thresholds,
    flags_from_thresholds,
    dwell_on_pattern_only,
)
from src.thresholding.hybrid_labelling import hybrid_from_flags


def _log(msg: str) -> None:
    print(f"[blend] {msg}")


def _fmt_window(h: int) -> str:
    d = h / 24
    return f"{d:.1f}d ({h}h)"


def _selected_combo_ids(all_combos: dict) -> list[str]:
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


def main() -> None:
    csv_in = os.path.join(config.RESULTS_RUN_DIR, "scores_with_thresholds.csv")
    sidecar_path = os.path.join(config.RESULTS_RUN_DIR, "columns_map.json")
    if not os.path.exists(csv_in) or not os.path.exists(sidecar_path):
        raise FileNotFoundError("Missing inputs. Run 01_generate_thresholds.py first.")

    df = pd.read_csv(csv_in, parse_dates=["timestamp"], low_memory=False)
    with open(sidecar_path, "r", encoding="utf-8") as f:
        sidecar = json.load(f)

    all_combos: dict = sidecar.get("combos", {})
    if not all_combos:
        _log("No combos found in sidecar; nothing to do.")
        return

    if not config.BLEND_SPECS:
        _log("No BLEND_SPECS configured; nothing to do.")
        csv_out = os.path.join(config.RESULTS_RUN_DIR, "scores_with_blends.csv")
        write_csv(df, csv_out, log_prefix="[blend]")
        return

    total = len(all_combos)
    selected = _selected_combo_ids(all_combos)
    scope = sidecar.get("meta", {}).get("combos_scope", "all")
    _log(f"Found {total} combos in sidecar (scope={scope}); {len(selected)} in SELECTED_COMBOS.")

    for (w_if, q_if, w_ae, q_ae) in config.SELECTED_COMBOS or []:
        _log(f"  Combo: IF={_fmt_window(w_if)} @ q={q_if}, AE={_fmt_window(w_ae)} @ q={q_ae}")

    sidecar.setdefault("blends", {})

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
                if_thr_adapt = meta["if_thresh"]          # e.g. if_adaptive_thresh__w1080_q030
                ae_thr_adapt = meta["ae_thresh"]          # e.g. lstm_adaptive_thresh__w1080_q975

                # 1) thresholds -> blended thresholds (capped min–max)
                thr_cols = capped_minmax_thresholds(
                    df,
                    if_adaptive_thr_col=if_thr_adapt,
                    ae_adaptive_thr_col=ae_thr_adapt,
                    static_if_thr=config.STATIC_THRESH_IF,
                    static_ae_thr=config.STATIC_THRESH_AE,
                    cap_delta=cap,
                )

                # 2) blended thresholds -> blended flags (nullable boolean), names derived canonically
                flag_cols = flags_from_thresholds(
                    df,
                    if_score_col="if_score",
                    ae_score_col="lstm_score",
                    if_thr_col=thr_cols["if_blend_thresh"],
                    ae_thr_col=thr_cols["ae_blend_thresh"],
                )

                # 3) hybrid label for the blend stage
                tag = N.tag(meta["if_window"], meta["if_quantile"], meta["ae_window"], meta["ae_quantile"])
                hybrid_col = N.hybrid_name(f"blend_cap{int(cap*1000):03d}", tag)
                df[hybrid_col] = hybrid_from_flags(
                    df[flag_cols["if_flag_col"]].astype("boolean"),
                    df[flag_cols["ae_flag_col"]].astype("boolean"),
                )

                # Update sidecar
                sidecar["blends"][name][cid] = {
                    "if_blend_thresh": thr_cols["if_blend_thresh"],
                    "ae_blend_thresh": thr_cols["ae_blend_thresh"],
                    "if_blend_flag": flag_cols["if_flag_col"],
                    "ae_blend_flag": flag_cols["ae_flag_col"],
                    "hybrid_label": hybrid_col,
                    # keep references to base adaptive flags too
                    "if_flag": meta.get("if_flag"),
                    "ae_flag": meta.get("ae_flag"),
                }

        elif op == "dwell_pattern":
            k = int(spec.get("k", 3))
            sidecar["blends"].setdefault(name, {})

            for cid in cids:
                meta = all_combos[cid]

                # Prefer most recent blended flags (from CAPPED_minmax) if present, else adaptive
                prev_blend = sidecar["blends"].get("CAPPED_minmax", {}).get(cid, {})
                if_flag = prev_blend.get("if_blend_flag", meta.get("if_flag"))
                ae_flag = prev_blend.get("ae_blend_flag", meta.get("ae_flag"))

                # Apply dwell only to AE; IF unchanged. (No combo_tag arg — the function derives it.)
                dwell_cols = dwell_on_pattern_only(
                    df,
                    if_flag_col=if_flag,
                    ae_flag_col=ae_flag,
                    k=k,
                )

                sidecar["blends"][name][cid] = {
                    "if_flag": dwell_cols["if_flag_col"],          # unchanged IF
                    "ae_flag_dwell": dwell_cols["ae_dwell_col"],   # AE after dwell
                    "hybrid_label": dwell_cols["hybrid_col"],
                }

        else:
            raise ValueError(f"Unsupported blend op: {op}")

    # Save outputs
    csv_out = os.path.join(config.RESULTS_RUN_DIR, "scores_with_blends.csv")
    write_csv(df, csv_out, log_prefix="[blend]")

    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)
    _log(f"Updated sidecar -> {sidecar_path}")


if __name__ == "__main__":
    main()
