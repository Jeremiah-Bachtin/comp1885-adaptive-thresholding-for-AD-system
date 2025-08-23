# src/utils/__init__.py
"""
Utility package for COMP1885.

Re-exports:
- Generic data helpers (data_utils)
- Evaluation window helpers (windows_utils)
- Run context helpers for deterministic per-run folders (run_context)
"""

from .data_utils import (
    load_scores,
    slice_by_date,
    mask_valid_rows,
    require_columns,
    ensure_nullable_bools,
    write_csv,
    ensure_static_flags,
)


from .run_context import (
    compute_slug,
    results_dir,
    write_readme,
    write_latest_pointer,
)

__all__ = [
    # data_utils
    "load_scores",
    "slice_by_date",
    "mask_valid_rows",
    "require_columns",
    "ensure_nullable_bools",
    "write_csv",
    "ensure_static_flags",
    # run_context
    "compute_slug",
    "results_dir",
    "write_readme",
    "write_latest_pointer",
]
