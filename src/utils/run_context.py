# src/utils/run_context.py
# Lightweight helpers for per-run context / output scoping.

from __future__ import annotations
import hashlib
import json
import os
from typing import Any, Dict, Iterable

# ----------------------------
# Slug / path helpers
# ----------------------------

def _compact_json(obj: Any) -> str:
    """
    Stable, whitespace-free JSON for hashing.
    Sorts keys to ensure identical payloads hash identically.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))

def compute_slug(fingerprint: Dict[str, Any]) -> str:
    """
    Compute an 8-char fingerprint for a dict of config knobs that
    materially affect outputs (windows, percentiles, blends, etc.).
    """
    payload = _compact_json(_json_safe(fingerprint))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]

def results_dir(base_dir: str, slug: str) -> str:
    """
    Return the per-run results directory path (no side effects).
    """
    return os.path.join(base_dir, slug)

# ----------------------------
# README + pointer helpers
# ----------------------------

def write_latest_pointer(interim_root: str, slug: str, filename: str = "LATEST_RUN.txt") -> None:
    """
    Write/refresh a plain-text pointer file at results/interim/LATEST_RUN.txt
    containing the current run slug.
    """
    os.makedirs(interim_root, exist_ok=True)
    ptr_path = os.path.join(interim_root, filename)
    with open(ptr_path, "w", encoding="utf-8") as f:
        f.write(slug.strip() + "\n")

def write_readme(run_dir: str, slug: str, summary: Dict[str, Any], filename: str = "README.txt") -> None:
    """
    Create/update a simple human-readable README with the key parameters that
    affect outputs. Idempotent; overwrites in place.
    """
    os.makedirs(run_dir, exist_ok=True)
    readme_path = os.path.join(run_dir, filename)
    text = _format_readme(slug, summary)
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(text)

def _format_readme(slug: str, summary: Dict[str, Any]) -> str:
    """
    Render a tidy README.txt showing the run slug and selected config values.
    """
    lines = [
        "=== COMP1885 Adaptive Thresholding Run ===",
        f"Run slug: {slug}",
        "",
        "Parameters:",
    ]
    for key, val in _sorted_items(summary):
        lines.append(f"- {key}: {_pretty(val)}")
    lines.append("")
    return "\n".join(lines)

# ----------------------------
# Utilities (pretty / JSON-safe)
# ----------------------------

def _json_safe(obj: Any) -> Any:
    """
    Convert common non-JSON-serialisable types (e.g. sets, tuples) into
    stable JSON-friendly forms for hashing / persistence.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, set):
        return sorted(_json_safe(x) for x in obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    # Fallback to string (stable repr) for odd types
    return str(obj)

def _pretty(val: Any) -> str:
    """
    Human-friendly, compact representation for README lines.
    """
    try:
        # Keep lists/tuples short but readable
        if isinstance(val, (list, tuple)):
            return "[" + ", ".join(_pretty(v) for v in val) + "]"
        if isinstance(val, dict):
            parts = [f"{k}={_pretty(v)}" for k, v in _sorted_items(val)]
            return "{" + ", ".join(parts) + "}"
        if isinstance(val, float):
            # Avoid excessive precision, but keep meaning (e.g. 0.975)
            return f"{val:.6g}"
        return str(val)
    except Exception:
        return repr(val)

def _sorted_items(d: Dict[str, Any]) -> Iterable[tuple[str, Any]]:
    """
    Deterministic ordering for README display & hashing stability.
    """
    return sorted(d.items(), key=lambda kv: kv[0].lower())
