# config/config.py

import os
from dotenv import load_dotenv
import re

# --- Project Root ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, "config/.env"), override=True)
from src.utils.run_context import compute_slug, results_dir, write_readme, write_latest_pointer

# ===============================
# Data & Model Paths
# ===============================
_data_path_env = os.getenv("DATA_PATH", os.path.join(PROJECT_ROOT, "data", "comp1884_train_scores.csv"))
DATA_PATH = os.path.join(PROJECT_ROOT, _data_path_env) if not os.path.isabs(_data_path_env) else _data_path_env

IF_MODEL_PATH = os.getenv("IF_MODEL_PATH", os.path.join(PROJECT_ROOT, "models", "if_model.joblib"))
LSTM_MODEL_PATH = os.getenv("LSTM_MODEL_PATH", os.path.join(PROJECT_ROOT, "models", "lstm_ae_best.h5"))

# ===============================
# Static thresholds
# ===============================
STATIC_THRESH_IF = float(os.getenv("STATIC_THRESH_IF", 0.03304385848702787))
STATIC_THRESH_AE = float(os.getenv("STATIC_THRESH_AE", 0.6425))
STATIC_THRESH_QUANT_IF = float(os.getenv("STATIC_THRESH_QUANT_IF", 0.03))
STATIC_THRESH_QUANT_AE = float(os.getenv("STATIC_THRESH_QUANT_AE", 0.95))
STATIC_IF_COL = os.getenv("STATIC_IF_COL", "is_if_anomaly").strip()
STATIC_AE_COL = os.getenv("STATIC_AE_COL", "is_lstm_anomaly").strip()

# ===============================
# Adaptive thresholds
# ===============================

# Default unit for bare numbers in window lists (days|hours)
DEFAULT_UNIT = os.getenv("COMBO_WINDOW_UNITS", "hours").strip().lower()

def _parse_window_token(tok: str, default_unit: str) -> int:
    tok = tok.strip().lower()
    m = re.fullmatch(r'(\d+)([dh])?', tok)
    if not m:
        raise ValueError(f"Bad window token: {tok}")
    n = int(m.group(1))
    suffix = m.group(2)
    if suffix == 'd': return n * 24
    if suffix == 'h': return n
    return n * 24 if default_unit == 'days' else n

def _parse_window_list(env_key: str, default: str) -> list[int]:
    raw = os.getenv(env_key, default)
    return [_parse_window_token(x, DEFAULT_UNIT) for x in raw.split(",") if x.strip()]

WINDOW_IF_LIST = _parse_window_list("WINDOW_IF_LIST", "14d")
WINDOW_AE_LIST = _parse_window_list("WINDOW_AE_LIST", "30d")
PERCENTILES_IF = [float(x) for x in os.getenv("PERCENTILES_IF", "0.03").split(",") if x.strip()]
PERCENTILES_AE = [float(x) for x in os.getenv("PERCENTILES_AE", "0.95").split(",") if x.strip()]

# Rolling warm-up behavior
ADAPTIVE_MIN_PERIODS = os.getenv("ADAPTIVE_MIN_PERIODS", "window").strip().lower()
if ADAPTIVE_MIN_PERIODS not in {"window", "1"}:
    ADAPTIVE_MIN_PERIODS = "window"

# ===============================
# Global scope
# ===============================
COMBOS_SCOPE = os.getenv("COMBOS_SCOPE", "all").strip().lower()
if COMBOS_SCOPE not in {"all", "selected"}:
    COMBOS_SCOPE = "all"

_selected_combos_env = os.getenv("SELECTED_COMBOS", "").strip()

def _parse_selected_combos(env_str: str, default_unit: str):
    out = []
    for part in env_str.split(","):
        if not part.strip(): continue
        w_if_str, q_if_str, w_ae_str, q_ae_str = [p.strip() for p in part.split(":")]
        w_if_h = _parse_window_token(w_if_str, default_unit)
        w_ae_h = _parse_window_token(w_ae_str, default_unit)
        out.append((w_if_h, float(q_if_str), w_ae_h, float(q_ae_str)))
    return out

SELECTED_COMBOS = _parse_selected_combos(_selected_combos_env, DEFAULT_UNIT) if _selected_combos_env else None

FULL_COMBOS_GRID = [
    (w_if, q_if, w_ae, q_ae)
    for w_if in WINDOW_IF_LIST
    for q_if in PERCENTILES_IF
    for w_ae in WINDOW_AE_LIST
    for q_ae in PERCENTILES_AE
]

def selected_combo_tuples() -> list[tuple[int, float, int, float]]:
    return SELECTED_COMBOS or []

def combos_scope() -> str:
    return COMBOS_SCOPE

def combos_to_build() -> list[tuple[int, float, int, float]]:
    return FULL_COMBOS_GRID if COMBOS_SCOPE == "all" else selected_combo_tuples()

MAX_WINDOW_HOURS = max((WINDOW_IF_LIST or [0]) + (WINDOW_AE_LIST or [0]))

# ===============================
# Evaluation windows
# ===============================
def _parse_windows(s: str):
    out = []
    if not s: return out
    for item in s.split(","):
        label, start, end = item.split("|")
        out.append({"label": label.strip(), "start": start.strip(), "end": end.strip()})
    return out

EVAL_WINDOWS = _parse_windows(os.getenv("EVAL_WINDOWS", "").strip())

# ===============================
# Results paths (root)
# ===============================
RESULTS_INTERIM_DIR = os.path.join(PROJECT_ROOT, "results", "interim")
RESULTS_FINAL_DIR = os.path.join(PROJECT_ROOT, "results", "final")
os.makedirs(RESULTS_INTERIM_DIR, exist_ok=True)
os.makedirs(RESULTS_FINAL_DIR, exist_ok=True)

# ===============================
# Blends
# ===============================
def _parse_blend_spec(raw: str) -> dict:
    spec: dict = {}
    for frag in (x.strip() for x in raw.split(";")):
        if not frag: continue
        if ":" in frag:
            key, val = frag.split(":", 1)
            key, val = key.strip(), val.strip()
        else:
            key, val = frag.strip(), ""
        if key == "k":
            spec[key] = int(val) if val else None
        elif key == "cap":
            spec[key] = float(val) if val else None
        else:
            spec[key] = val
    return spec

_blends_env = os.getenv("BLENDS", "").split(",")
BLEND_SPECS = {}
for name in (b.strip() for b in _blends_env if b.strip()):
    raw = os.getenv(f"BLEND_{name}", "")
    if raw:
        BLEND_SPECS[name] = {"name": name, **_parse_blend_spec(raw)}

# ===============================
# Legacy / compatibility (counts scope)
# ===============================
LEGACY_COUNT_COMBOS_SCOPE = os.getenv("COUNT_COMBOS_SCOPE", "").strip().lower() or None
if LEGACY_COUNT_COMBOS_SCOPE and LEGACY_COUNT_COMBOS_SCOPE not in {"selected", "all"}:
    LEGACY_COUNT_COMBOS_SCOPE = None

# ===============================
# Confidence Scoring (percentile-rank)
# ===============================
def _get_bool(env_key: str, default: str | int = "1") -> bool:
    v = os.getenv(env_key, str(default)).strip().lower()
    return v in {"1", "true", "yes", "on", "y", "t"}

def _get_float(env_key: str, default: float) -> float:
    try:
        return float(os.getenv(env_key, str(default)))
    except Exception:
        return default

CONF_ENABLED = _get_bool("CONF_ENABLED", "1")
CONF_METHOD = os.getenv("CONF_METHOD", "percentile_rank").strip().lower()
if CONF_METHOD not in {"percentile_rank", "zscore"}:
    CONF_METHOD = "percentile_rank"

CONF_TAIL_IF = os.getenv("CONF_TAIL_IF", "low").strip().lower()
CONF_TAIL_AE = os.getenv("CONF_TAIL_AE", "high").strip().lower()
if CONF_TAIL_IF not in {"low", "high"}:  CONF_TAIL_IF = "low"
if CONF_TAIL_AE not in {"low", "high"}:  CONF_TAIL_AE = "high"

CONF_MIN_PERIODS_POLICY = os.getenv("CONF_MIN_PERIODS", "inherit").strip().lower()
if CONF_MIN_PERIODS_POLICY not in {"inherit", "window", "1"}:
    CONF_MIN_PERIODS_POLICY = "inherit"

CONF_WINDOW_SOURCE = os.getenv("CONF_WINDOW_SOURCE", "combo").strip().lower()
if CONF_WINDOW_SOURCE not in {"combo"}:
    CONF_WINDOW_SOURCE = "combo"

CONF_EPS       = _get_float("CONF_EPS", 1e-9)
CONF_CLIP_MIN  = _get_float("CONF_CLIP_MIN", 1e-3)
CONF_CLIP_MAX  = _get_float("CONF_CLIP_MAX", 0.999)

_raw_emit = os.getenv("CONF_EMIT_TYPES", "Point,Pattern,Compound")
CONF_EMIT_TYPES = {x.strip() for x in _raw_emit.split(",") if x.strip()}

CONF_COMPOUND_AGG  = os.getenv("CONF_COMPOUND_AGG", "mean").strip().lower()
CONF_POINT_SOURCE  = os.getenv("CONF_POINT_SOURCE", "IF").strip().upper()
CONF_PATTERN_SOURCE= os.getenv("CONF_PATTERN_SOURCE", "AE").strip().upper()

CONF_SCOPE_RESPECTS_COMBOS = _get_bool("CONF_SCOPE_RESPECTS_COMBOS", "1")
CONF_COL_PREFIX = os.getenv("CONF_COL_PREFIX", "conf").strip()

# Implementation toggle — include in slug
CONF_IMPL = os.getenv("CONF_IMPL", "combo").strip().lower()  # 'combo' or 'qt'
if CONF_IMPL not in {"combo", "qt"}:
    CONF_IMPL = "combo"

def resolve_min_periods(window_hours: int) -> int:
    policy = CONF_MIN_PERIODS_POLICY
    if policy == "inherit":
        policy = ADAPTIVE_MIN_PERIODS
    return window_hours if policy == "window" else 1

def tail_direction_for(model: str) -> str:
    m = model.strip().upper()
    return CONF_TAIL_IF if m == "IF" else CONF_TAIL_AE

def should_emit_for_label(label: str) -> bool:
    return (label or "").strip().title() in CONF_EMIT_TYPES

# =====================================================================
# Build fingerprint and run slug
# =====================================================================

# 1) Fingerprint for slug (ONLY knobs that define outputs/behaviour)
FINGERPRINT = {
    # --- adaptive / combos ---
    "WINDOW_IF_LIST": WINDOW_IF_LIST,
    "WINDOW_AE_LIST": WINDOW_AE_LIST,
    "PERCENTILES_IF": PERCENTILES_IF,
    "PERCENTILES_AE": PERCENTILES_AE,
    "COMBOS_SCOPE": COMBOS_SCOPE,
    "SELECTED_COMBOS": SELECTED_COMBOS,            # may be None or list[tuple]
    "ADAPTIVE_MIN_PERIODS": ADAPTIVE_MIN_PERIODS,

    # --- eval / blends / static baselines ---
    "EVAL_WINDOWS": EVAL_WINDOWS,
    "BLEND_SPECS": BLEND_SPECS,                    # dict of named blend specs
    "STATIC_THRESH_IF": STATIC_THRESH_IF,
    "STATIC_THRESH_AE": STATIC_THRESH_AE,
    "STATIC_THRESH_QUANT_IF": STATIC_THRESH_QUANT_IF,
    "STATIC_THRESH_QUANT_AE": STATIC_THRESH_QUANT_AE,

    # --- confidence (ALL knobs together, incl. method switch) ---
    "CONF_ENABLED": CONF_ENABLED,
    "CONF_METHOD": CONF_METHOD,
    "CONF_TAIL_IF": CONF_TAIL_IF,
    "CONF_TAIL_AE": CONF_TAIL_AE,
    "CONF_MIN_PERIODS_POLICY": CONF_MIN_PERIODS_POLICY,
    "CONF_WINDOW_SOURCE": CONF_WINDOW_SOURCE,
    "CONF_EPS": CONF_EPS,
    "CONF_CLIP_MIN": CONF_CLIP_MIN,
    "CONF_CLIP_MAX": CONF_CLIP_MAX,
    "CONF_EMIT_TYPES": sorted(CONF_EMIT_TYPES),    # normalise set -> sorted list
    "CONF_COMPOUND_AGG": CONF_COMPOUND_AGG,
    "CONF_POINT_SOURCE": CONF_POINT_SOURCE,
    "CONF_PATTERN_SOURCE": CONF_PATTERN_SOURCE,
    "CONF_SCOPE_RESPECTS_COMBOS": CONF_SCOPE_RESPECTS_COMBOS,
    "CONF_COL_PREFIX": CONF_COL_PREFIX,
    "CONF_IMPL": CONF_IMPL,                        # <-- the knob that switches combo vs qt
}

# 2) README summary (human-friendly; can include extras like DATA_PATH)
README_SUMMARY = {
    # context
    "DATA_PATH": DATA_PATH,
    "MAX_WINDOW_HOURS": MAX_WINDOW_HOURS,

    # adaptive / combos
    "COMBOS_SCOPE": COMBOS_SCOPE,
    "SELECTED_COMBOS": SELECTED_COMBOS,
    "WINDOW_IF_LIST": WINDOW_IF_LIST,
    "WINDOW_AE_LIST": WINDOW_AE_LIST,
    "PERCENTILES_IF": PERCENTILES_IF,
    "PERCENTILES_AE": PERCENTILES_AE,
    "ADAPTIVE_MIN_PERIODS": ADAPTIVE_MIN_PERIODS,

    # eval / blends / static
    "EVAL_WINDOWS": EVAL_WINDOWS,
    "BLEND_SPECS": BLEND_SPECS,
    "STATIC_THRESH_IF": STATIC_THRESH_IF,
    "STATIC_THRESH_AE": STATIC_THRESH_AE,
    "STATIC_IF_COL": STATIC_IF_COL,
    "STATIC_AE_COL": STATIC_AE_COL,

    # confidence
    "CONF_ENABLED": CONF_ENABLED,
    "CONF_METHOD": CONF_METHOD,
    "CONF_TAIL_IF": CONF_TAIL_IF,
    "CONF_TAIL_AE": CONF_TAIL_AE,
    "CONF_MIN_PERIODS_POLICY": CONF_MIN_PERIODS_POLICY,
    "CONF_WINDOW_SOURCE": CONF_WINDOW_SOURCE,
    "CONF_EMIT_TYPES": sorted(CONF_EMIT_TYPES),
    "CONF_COMPOUND_AGG": CONF_COMPOUND_AGG,
    "CONF_POINT_SOURCE": CONF_POINT_SOURCE,
    "CONF_PATTERN_SOURCE": CONF_PATTERN_SOURCE,
    "CONF_SCOPE_RESPECTS_COMBOS": CONF_SCOPE_RESPECTS_COMBOS,
    "CONF_COL_PREFIX": CONF_COL_PREFIX,
    "CONF_IMPL": CONF_IMPL,
}

# 3) Compute slug & make results dir AFTER everything above is defined
RUN_SLUG = compute_slug(FINGERPRINT)
RESULTS_RUN_DIR = results_dir(RESULTS_INTERIM_DIR, RUN_SLUG)
os.makedirs(RESULTS_RUN_DIR, exist_ok=True)

# 4) Write README for convenience
write_readme(RESULTS_RUN_DIR, RUN_SLUG, summary=README_SUMMARY)