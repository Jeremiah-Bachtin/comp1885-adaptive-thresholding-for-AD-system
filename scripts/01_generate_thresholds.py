# scripts/01_generate_thresholds.py
from __future__ import annotations
import os, json
import config.config as config
from src.utils import load_scores, write_csv, require_columns, ensure_static_flags
from src.thresholding.ids import combo_id
from src.thresholding.rolling import compute_and_flag
# naming helpers
from src.utils import naming as N

def _log(msg: str) -> None:
    print(f"[thresholds] {msg}")

def main() -> None:
    # 1) Load
    df = load_scores(config.DATA_PATH)
    require_columns(df, ["timestamp", "if_score", "lstm_score"], context="scores dataset")
    _log(f"Loaded dataset: {config.DATA_PATH}  (rows={len(df):,})")

    # 2) Combos by scope
    combos = list(config.combos_to_build())
    if not combos:
        raise RuntimeError("No combos to build. If COMBOS_SCOPE=selected, ensure SELECTED_COMBOS is set.")
    _log(f"Preparing {len(combos)} combos (scope={config.combos_scope()}).")

    unique_if = sorted({(w_if, q_if) for (w_if, q_if, _, _) in combos})
    unique_ae = sorted({(w_ae, q_ae) for (_, _, w_ae, q_ae) in combos})

    # 3) Sidecar
    sidecar: dict = {
        "meta": {"combos_scope": config.combos_scope(), "n_combos": len(combos)},
        "if_settings": [],
        "ae_settings": [],
        "combos": {},
        "naming": {"style": "prefix__w<hrs>_q<3dig>", "tag_example": N.tag(1080, 0.03, 1080, 0.975)},
    }

    # 4) Warm‑up policy
    minp = None if config.ADAPTIVE_MIN_PERIODS == "window" else 1

    # 5) IF thresholds/flags
    for w_if, q_if in unique_if:
        tcol = N.thr_if_adapt(w_if, q_if)
        fcol = N.flg_if_adapt(w_if, q_if)
        compute_and_flag(df, "if_score", w_if, q_if, "if", direction="low", min_periods=minp)
        sidecar["if_settings"].append({"window": w_if, "quantile": q_if, "thresh_col": tcol, "flag_col": fcol})
    _log(f"Computed IF thresholds & flags for {len(unique_if)} (w,q) pairs.")

    # 6) AE thresholds/flags
    for w_ae, q_ae in unique_ae:
        tcol = N.thr_ae_adapt(w_ae, q_ae)
        fcol = N.flg_ae_adapt(w_ae, q_ae)
        compute_and_flag(df, "lstm_score", w_ae, q_ae, "lstm", direction="high", min_periods=minp)
        sidecar["ae_settings"].append({"window": w_ae, "quantile": q_ae, "thresh_col": tcol, "flag_col": fcol})
    _log(f"Computed AE thresholds & flags for {len(unique_ae)} (w,q) pairs.")

    # 7) Hybrid labels per combo (adaptive)
    new_hybrids = 0
    for (w_if, q_if, w_ae, q_ae) in combos:
        if_flag = N.flg_if_adapt(w_if, q_if)
        ae_flag = N.flg_ae_adapt(w_ae, q_ae)
        cid = combo_id(w_if, q_if, w_ae, q_ae)
        tag = N.tag(w_if, q_if, w_ae, q_ae)
        lbl = N.hybrid_name("adaptive", tag)

        from src.thresholding.hybrid_labelling import hybrid_from_flags
        df[lbl] = hybrid_from_flags(df[if_flag].astype("boolean"), df[ae_flag].astype("boolean"))
        new_hybrids += 1

        sidecar["combos"][cid] = {
            "if_window": w_if, "if_quantile": q_if, "ae_window": w_ae, "ae_quantile": q_ae,
            "if_thresh": N.thr_if_adapt(w_if, q_if), "if_flag": if_flag,
            "ae_thresh": N.thr_ae_adapt(w_ae, q_ae), "ae_flag": ae_flag,
            "hybrid_label": lbl,
        }
    _log(f"Built hybrid labels for {new_hybrids} combos.")

    # 8) Static baseline flags
    created = ensure_static_flags(
        df,
        if_score_col="if_score", ae_score_col="lstm_score",
        static_if_thr=config.STATIC_THRESH_IF, static_ae_thr=config.STATIC_THRESH_AE,
        static_if_col=config.STATIC_IF_COL, static_ae_col=config.STATIC_AE_COL,
        force_recreate=False,
    )
    if created:
        _log(f"Created static flags: {created}")

    # 9) Save
    out_csv = os.path.join(config.RESULTS_RUN_DIR, "scores_with_thresholds.csv")
    write_csv(df, out_csv, log_prefix="[thresholds]")
    sidecar_path = os.path.join(config.RESULTS_RUN_DIR, "columns_map.json")
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)
    _log(f"Saved sidecar -> {sidecar_path}")
    _log("Done.")

if __name__ == "__main__":
    main()
