# src/utils/naming.py

from src.thresholding.ids import q_to_str

def tag(w_if_h:int, q_if:float, w_ae_h:int, q_ae:float) -> str:
    return f"wIF{int(w_if_h)}h_qIF{q_to_str(q_if)}__wAE{int(w_ae_h)}h_qAE{q_to_str(q_ae)}"

def col(prefix:str, tag:str) -> str:
    # generic “prefix__{tag}” helper so we stop re-typing patterns
    return f"{prefix}__{tag}"

# Consistent primitives used everywhere:
def thr_if_adapt(w_if_h,q_if):   return col("if_adaptive_thresh", f"w{int(w_if_h)}_q{q_to_str(q_if)}")
def flg_if_adapt(w_if_h,q_if):   return col("is_if_adaptive",     f"w{int(w_if_h)}_q{q_to_str(q_if)}")
def thr_ae_adapt(w_ae_h,q_ae):   return col("lstm_adaptive_thresh", f"w{int(w_ae_h)}_q{q_to_str(q_ae)}")
def flg_ae_adapt(w_ae_h,q_ae):   return col("is_lstm_adaptive",     f"w{int(w_ae_h)}_q{q_to_str(q_ae)}")

def thr_if_blend(base, cap):     return f"{base}_blend_cap{int(cap*1000):03d}"
def flg_if_blend(base, cap):     return f"{base}_blend_cap{int(cap*1000):03d}"

def dwell(col, k):               return f"{col}_dwell{k}"

def hybrid_name(kind:str, tag:str) -> str:   # kind: 'adaptive' | 'blend_cap020' | 'pattern_dwell3' etc.
    return f"hybrid_label_{kind}__{tag}"

def conf_if(tag:str) -> str:     return col("conf_if", tag)
def conf_ae(tag:str) -> str:     return col("conf_ae", tag)
def conf_final(tag:str, variant:str) -> str: # per-variant final emission
    return col("conf", f"{variant}__{tag}")
