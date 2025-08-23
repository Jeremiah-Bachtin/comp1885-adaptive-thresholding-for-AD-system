# src/utils/naming.py

from __future__ import annotations
from typing import Optional
from src.thresholding.ids import q_to_str

# --- canonical combo tag -----------------------------------------------------
def tag(w_if_h: int, q_if: float, w_ae_h: int, q_ae: float) -> str:
    return f"wIF{int(w_if_h)}h_qIF{q_to_str(q_if)}__wAE{int(w_ae_h)}h_qAE{q_to_str(q_ae)}"

def _maybe_suffix(base: str, combo_tag: Optional[str]) -> str:
    return f"{base}__{combo_tag}" if combo_tag else base

# --- generic prefix+tag helper (kept for convenience/compat) -----------------
def col(prefix: str, tag_str: str) -> str:
    return f"{prefix}__{tag_str}"

# --- ADAPTIVE primitives (optionally suffix with full combo tag) -------------
def thr_if_adapt(w_if_h: int, q_if: float, combo_tag: Optional[str] = None) -> str:
    base = f"if_adaptive_thresh__w{int(w_if_h)}_q{q_to_str(q_if)}"
    return _maybe_suffix(base, combo_tag)

def flg_if_adapt(w_if_h: int, q_if: float, combo_tag: Optional[str] = None) -> str:
    base = f"is_if_adaptive__w{int(w_if_h)}_q{q_to_str(q_if)}"
    return _maybe_suffix(base, combo_tag)

def thr_ae_adapt(w_ae_h: int, q_ae: float, combo_tag: Optional[str] = None) -> str:
    base = f"lstm_adaptive_thresh__w{int(w_ae_h)}_q{q_to_str(q_ae)}"
    return _maybe_suffix(base, combo_tag)

def flg_ae_adapt(w_ae_h: int, q_ae: float, combo_tag: Optional[str] = None) -> str:
    base = f"is_lstm_adaptive__w{int(w_ae_h)}_q{q_to_str(q_ae)}"
    return _maybe_suffix(base, combo_tag)

# --- BLEND suffixing ---------------------------------------------------------
def thr_if_blend(base_thresh_col: str, cap: float) -> str:
    # works for IF/AE thresholds; just appends the cap suffix
    return f"{base_thresh_col}_blend_cap{int(cap * 1000):03d}"

def flg_if_blend(base_flag_col: str, cap: float) -> str:
    # NOTE: flags get '_blend_flag_' (not just '_blend_')
    return f"{base_flag_col}_blend_flag_cap{int(cap * 1000):03d}"

# --- DWELL suffixing ---------------------------------------------------------
def dwell(colname: str, k: int) -> str:
    return f"{colname}_dwell{k}"

# --- HYBRID labels -----------------------------------------------------------
def hybrid_name(kind: str, combo_tag: str) -> str:
    # kind: 'adaptive' | 'blend_cap020' | 'pattern_dwell3' | etc.
    return f"hybrid_label_{kind}__{combo_tag}"

# --- Confidence columns ------------------------------------------------------
def conf_if(combo_tag: str) -> str:
    return col("conf_if", combo_tag)

def conf_ae(combo_tag: str) -> str:
    return col("conf_ae", combo_tag)

def conf_final(combo_tag: str, variant: str) -> str:
    # final per‑timestamp emission for a given variant (adaptive/blend/dwell)
    return col("conf", f"{variant}__{combo_tag}")
