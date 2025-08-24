# `src/thresholding`

Core logic for building **adaptive thresholds**, **blends**, and **hybrid anomaly labels**.  
All functions are small, explicit, and work on Pandas DataFrames.

---

## What lives here

- **`rolling.py`** — compute rolling quantile thresholds and flags.
- **`blending.py`** — apply capped min–max blending and AE dwell; rebuild flags + labels.
- **`hybrid_labelling.py`** — turn IF/AE flags into one label: `Point`, `Pattern`, `Compound`, or `None`.
- **`ids.py`** — stable IDs for combos; quantile → 3‑digit text.
- **`counting.py`** — tidy count tables and comparisons vs static.

> Column names are created via helpers in `src/utils/naming.py`.  
> Scripts write/read a **sidecar** (`columns_map.json`) describing columns per combo.

---

## Key concepts

### Scores and tails
- **IF** (Isolation Forest): **lower** scores = more anomalous → threshold is a **low‑tail** quantile.
- **AE/LSTM-AE**: **higher** errors = more anomalous → threshold is a **high‑tail** quantile.

### Combo (window + quantiles)
A **combo** is `(w_if_h, q_if, w_ae_h, q_ae)` in **hours** and **floats**.  
It has a canonical **tag**:
```
wIF{w_if}h_qIF{qqq}__wAE{w_ae}h_qAE{qqq}
# example: wIF1080h_qIF030__wAE1080h_qAE975
```

### Hybrid label (per row)
- `Point`   → IF flag = True, AE flag ≠ True  
- `Pattern` → AE flag = True, IF flag ≠ True  
- `Compound`→ IF flag = True and AE flag = True  
- `None`    → otherwise (or NA)

---

## File-by-file

### `rolling.py`
Build **adaptive** thresholds and flags.

- `compute_rolling_threshold(df, score_col, window, quantile, model_prefix, combo_tag, min_periods)`
  - Rolling quantile over a **trailing** window.
  - Writes a threshold column (named via `naming.*`).
- `apply_flag(df, score_col, threshold_col, direction, flag_col)`
  - Compares score to threshold (`low` or `high`).
  - Writes a **nullable boolean** flag.
- `compute_and_flag(...)`
  - Convenience: threshold + flag in one call.
- `make_colnames(...)`
  - Returns canonical names for a given combo.

**Expectations**
- `df[score_col]` is numeric.
- `min_periods`: use `window` (strict) or `1` (start from row 1) to match `.env`.

---

### `blending.py`
Refine thresholds/flags; update hybrid labels.

- `capped_minmax_thresholds(df, if_adaptive_thr_col, ae_adaptive_thr_col, static_if_thr, static_ae_thr, cap_delta)`
  - **IF**: `max( min(static_IF, IF_adapt), IF_adapt - δ )`
  - **AE**: `min( max(static_AE, AE_adapt), AE_adapt + δ )`
  - Writes `*_blend_cap{ddd}` thresholds; returns their names.
- `flags_from_thresholds(df, if_score_col, ae_score_col, if_thr_col, ae_thr_col)`
  - Builds blended flags named `*_blend_flag_cap{ddd}`; returns names.
- `dwell_on_pattern_only(df, if_flag_col, ae_flag_col, k)`
  - Apply **k‑consecutive** dwell to **AE only** (Pattern). IF is unchanged.
  - Writes `*_dwell{k}` and a new `hybrid_label_pattern_dwell{k}__{tag}`.

**Notes**
- Works with any combo; relies on canonical suffixes (`__w{h}_q{qqq}`).
- Output columns are added in place and returned as strings.

---

### `hybrid_labelling.py`
Single function:

- `hybrid_from_flags(if_flag: Series, ae_flag: Series) -> Series`
  - Returns the label series with dtype `string`.
  - Input flags should be **nullable boolean**.

---

### `ids.py`
Stable string IDs.

- `q_to_str(0.975) -> "975"`; `q_to_str(0.03) -> "030"`.
- `combo_id(w_if, q_if, w_ae, q_ae) -> "IFw{w_if}q{qqq}_AEw{w_ae}q{qqq}"`.
  - Used as keys in the sidecar and logs.

---

### `counting.py`
Tidy counts and adaptive vs static comparison.

- `count_from_label_series(label) -> DataFrame`
  - Rows in fixed order: `Point`, `Pattern`, `Compound`; zero‑filled if missing.
- `counts_variant_vs_static(dfw, if_col, ae_col, static_if_col, static_ae_col) -> DataFrame`
  - Builds labels from `(if_col, ae_col)` and from static columns.
  - Returns `anomaly_type, count_variant, count_static, valid_n`.

---

## Naming quick reference

All names come from `src/utils/naming.py`:

- Adaptive thresholds  
  - IF:   `if_adaptive_thresh__w{h}_q{qqq}__{tag}`  
  - AE:   `lstm_adaptive_thresh__w{h}_q{qqq}__{tag}`
- Adaptive flags  
  - IF:   `is_if_adaptive__w{h}_q{qqq}__{tag}`  
  - AE:   `is_lstm_adaptive__w{h}_q{qqq}__{tag}`
- Blended thresholds / flags  
  - `*_blend_cap{ddd}`, `*_blend_flag_cap{ddd}`
- Dwell flags  
  - `*_dwell{k}`
- Hybrid labels  
  - `hybrid_label_{variant}__{tag}`  
  - `variant ∈ {adaptive, blend_cap{ddd}, pattern_dwell{k}}`

---

## Typical flow (per combo)

1. **Adaptive:**  
   `compute_and_flag` on IF (`low`) and AE (`high`) → flags + `hybrid_label_adaptive__{tag}`
2. **Blend (optional):**  
   `capped_minmax_thresholds` → `flags_from_thresholds` → `hybrid_label_blend_cap{ddd}__{tag}`
3. **Dwell (optional):**  
   `dwell_on_pattern_only` → `hybrid_label_pattern_dwell{k}__{tag}`

Scripts will choose the **latest** label (Dwell → Blend → Adaptive) for counting.

---

## Inputs and dtypes (gotchas)

- Ensure score columns exist: `if_score`, `lstm_score`.
- Flags should be **nullable boolean** (`"boolean"` dtype).
- Rolling windows treat each row as **1 hour** (dataset is hourly).
- Warm‑up behaviour must match `.env` to align results (`min_periods`).

---

## Glossary

- **Point**: IF‑only anomaly (sudden spike in IF tail).  
- **Pattern**: AE‑only anomaly (sustained pattern in AE tail).  
- **Compound**: Both IF and AE agree.  
- **None**: Not flagged or insufficient data in window.
