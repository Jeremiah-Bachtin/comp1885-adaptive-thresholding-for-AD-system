# scripts/run_full_pipeline.py
"""
Run the COMP1885 pipeline end-to-end with deterministic, readable logs.

Order rationale
---------------
1) 01_generate_thresholds.py         -> compute adaptive quantiles (per combo, rolling)
2) 02_generate_blends.py             -> apply capped-minmax + dwell (final flags/labels)
3) 03_generate_counts.py             -> per-window counts vs static (diagnostics)
4) 04_generate_counts_global.py      -> aggregated/global summaries (optional analytics)
5) 05_compute_confidence_[combo|qt].py
                                      -> percentile-rank confidence using the SAME windows
                                         and warm-up as thresholds; uses final labels from (2)

Note:
- Steps (3) and (4) are independent of (5); (5) must come after (2).
- The run slug computed in config.py now includes CONF_IMPL, so switching
  between 'combo' and 'qt' always generates a new slug/output directory.
"""
from __future__ import annotations
import os
import sys
import time
import subprocess
from typing import Sequence, Tuple
import warnings
warnings.filterwarnings("ignore", category=getattr(__import__("pandas").errors, "PerformanceWarning"))
try:
    warnings.filterwarnings("ignore", category=__import__("pandas").core.common.PerformanceWarning)
except Exception:
    pass


# Config + run context
from config.config import RESULTS_RUN_DIR, RUN_SLUG, CONF_IMPL
from src.utils import write_latest_pointer

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

def _log(msg: str) -> None:
    print(msg, flush=True)

def _script_path(name: str) -> str:
    return os.path.join(SCRIPTS_DIR, name)

def run_step(name: str) -> int:
    """
    Execute a pipeline step as a separate Python process.
    Returns the process return code (0 = success).
    """
    path = _script_path(name)
    if not os.path.exists(path):
        _log(f"[runner][skip] {name} not found at {path}")
        return 0
    _log(f"[runner] → {name}")
    t0 = time.time()
    # inherit environment; each child imports config.py and sees the same CONF_IMPL
    proc = subprocess.run([sys.executable, path], cwd=SCRIPTS_DIR)
    dt = time.time() - t0
    _log(f"[runner] ← {name} (rc={proc.returncode}, {dt:.2f}s)")
    return proc.returncode

def main() -> None:
    impl = (CONF_IMPL or "combo").strip().lower()
    if impl not in {"combo", "qt"}:
        impl = "combo"
    conf_script = "05_compute_confidence_qt.py" if impl == "qt" else "05_compute_confidence_combo.py"

    STEPS: Sequence[Tuple[str, bool]] = (
        ("01_generate_thresholds.py",        True),
        ("02_generate_blends.py",            True),
        ("03_generate_counts.py",            True),
        ("04_generate_counts_global.py",     True),
        (conf_script,                        True),  # confidence last to use final labels
    )

    os.makedirs(RESULTS_RUN_DIR, exist_ok=True)

    _log("=== COMP1885 Pipeline ===")
    _log(f"[runner] Run slug: {RUN_SLUG}")
    _log(f"[runner] Results dir: {RESULTS_RUN_DIR}")
    _log(f"[runner] CONF_IMPL: {impl}")

    failures = 0
    for name, enabled in STEPS:
        if not enabled:
            _log(f"[runner][off] {name}")
            continue
        rc = run_step(name)
        if rc != 0:
            failures += 1
            _log(f"[runner][fail] {name} exited with {rc}. Stopping.")
            break

    if failures == 0:
        write_latest_pointer(os.path.dirname(RESULTS_RUN_DIR), RUN_SLUG)
        _log("[runner] All enabled steps finished successfully.")
    else:
        _log("[runner] Pipeline terminated due to errors.")

if __name__ == "__main__":
    main()
