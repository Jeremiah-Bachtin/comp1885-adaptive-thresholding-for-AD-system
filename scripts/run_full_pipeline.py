# scripts/run_full_pipeline.py
"""
Full pipeline runner for COMP1885 adaptive thresholding.

Steps:
  1) Generate thresholds & labels for combos
  2) Generate blends (if configured)
  3) Generate counts by evaluation windows
  4) Generate global counts summary

All outputs are written under the slugged folder:
  results/interim/<RUN_SLUG>/
"""

import subprocess
import sys
import os
import config.config as config
from src.utils.run_context import write_latest_pointer

LATEST_PTR = os.path.join(config.RESULTS_INTERIM_DIR, "LATEST_RUN.txt")


def _read_latest_slug() -> str | None:
    try:
        if os.path.exists(LATEST_PTR):
            with open(LATEST_PTR, "r", encoding="utf-8") as f:
                return f.read().strip()
    except Exception:
        pass
    return None


def _run(script_name: str) -> None:
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")

    print(f"[runner] Executing {script_name} ...")
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")
    if result.returncode != 0:
        raise SystemExit(f"[runner] Step failed: {script_name} (code {result.returncode})")
    print(f"[runner] Completed {script_name}\n")


def main() -> None:
    print("=== COMP1885 Adaptive Thresholding Pipeline ===")
    print(f"Dataset: {config.DATA_PATH}")
    print(f"Eval windows: {[w['label'] for w in config.EVAL_WINDOWS]}")
    print("----------------------------------------------")
    print(f"RUN_SLUG: {config.RUN_SLUG}")
    print(f"Output dir: {config.RESULTS_RUN_DIR}")

    last = _read_latest_slug()
    if last and last != config.RUN_SLUG:
        print(f"[runner] NOTE: Config fingerprint changed.")
        print(f"        Last run slug: {last}")
        print(f"        This run slug: {config.RUN_SLUG}")
    elif last == config.RUN_SLUG:
        print(f"[runner] NOTE: Same config as last run (slug={config.RUN_SLUG}); "
              f"artefacts will be overwritten.")
    else:
        print("[runner] First run (no LATEST_RUN.txt present).")
    print("==============================================\n")

    # Steps
    _run("01_generate_thresholds.py")
    _run("02_generate_blends.py")
    _run("03_generate_counts.py")
    _run("04_generate_counts_global.py")

    # Only update pointer if pipeline succeeds
    write_latest_pointer(config.RESULTS_INTERIM_DIR, config.RUN_SLUG)

    print("[runner] Pipeline completed successfully.")
    print(f"[runner] Outputs written under: {config.RESULTS_RUN_DIR}")
    print(f"[runner] Latest run pointer updated: {LATEST_PTR}")

    print("\n[runner] Key artefacts:")
    for fn in [
        "scores_with_thresholds.csv",
        "scores_with_blends.csv",
        "columns_map.json",
        "window_counts_detail.csv",
        "window_counts_totals.csv",
        "global_counts_summary.csv",
        "README.txt",
    ]:
        print(f" - {os.path.join(config.RESULTS_RUN_DIR, fn)}")


if __name__ == "__main__":
    main()
