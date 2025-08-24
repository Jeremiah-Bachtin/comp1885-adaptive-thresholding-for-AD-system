# `src/confidence_scoring`

Confidence scoring for anomaly labels using **rolling percentile ranks**.  
Two implementations share the same idea but differ in the reference quantile.

---

## Files

- **`percentile_rank_combo.py`** — confidence vs the **fixed combo quantiles** (`q_if`, `q_ae`).  
- **`percentile_rank_qt.py`** — confidence vs the **realised thresholds** over time (`θ_t`).

Both expose:  
`rolling_percentile_conf(scores, *, tail, window_hours, min_periods, eps, clip_min, clip_max, ...) -> Series`

---

## Core idea

For each timestamp, compute the **ECDF percentile** of the current score using a **trailing window** that **excludes** the current row.  
Then map that percentile to a **[0, 1] confidence** that grows as the point moves **deeper into the anomalous tail**.

- **IF** (low tail anomaly): lower scores are more anomalous.  
- **AE** (high tail anomaly): higher scores are more anomalous.

We keep settings aligned with thresholds (same window and warm‑up) so confidence is directly comparable to labels.

---

## Methods

### 1) `combo` (fixed‑q reference)
Used by `percentile_rank_combo.py`.

Let `F_t = ECDF_window(x_t)` and `q` be the combo’s quantile.

- **AE (tail='high')**: anomaly when `F_t ≥ q`  
  `conf = max(0, (F_t - q) / max(eps, 1 - q))`
- **IF (tail='low')**: anomaly when `F_t ≤ q`  
  `conf = max(0, (q - F_t) / max(eps, q))`

This gives ~0 near the threshold and →1 deeper into the tail.

### 2) `qt` (time‑varying reference)
Used by `percentile_rank_qt.py`.

We compute the ECDF position of the **realised threshold** at time `t`: `q*_t = ECDF_window(θ_t)`.  
Then apply the same mapping as above, but with `q*_t`.  
If `θ_t` is missing, we fall back to the fixed combo `q`.

---

## ECDF details

We use **Hazen** plotting positions:  
`ECDF(x) = (rank(x) - 0.5) / n`, clipped to `(eps, 1 - eps)` to avoid exact 0/1.  
Window uses `window_hours` rows; warm‑up controlled by `min_periods` and matches threshold settings.

---

## Config knobs (from `.env` via `config.py`)

- `CONF_ENABLED`: master switch.  
- `CONF_METHOD`: currently `percentile_rank`.  
- `CONF_TAIL_IF`, `CONF_TAIL_AE`: `"low"` or `"high"`.  
- `CONF_MIN_PERIODS`: `inherit | window | 1` (align with thresholds).  
- `CONF_WINDOW_SOURCE`: `combo` (use combo windows).  
- `CONF_EPS`, `CONF_CLIP_MIN`, `CONF_CLIP_MAX`: numeric stability and final clipping.  
- `CONF_EMIT_TYPES`: which labels to emit confidence for (`Point,Pattern,Compound`).  
- `CONF_COMPOUND_AGG`: how to combine IF/AE confidences for `Compound` (`mean|min|max`).  
- `CONF_POINT_SOURCE`, `CONF_PATTERN_SOURCE`: source confidence for single‑model labels.  
- `CONF_COL_PREFIX`: output prefix (`conf`).  
- `CONF_IMPL`: `combo` or `qt` (controls which script runs).

---

## Output columns

Final per‑timestamp emission (one column per **hybrid label column** found):  
```
conf__{variant}__{tag}
# examples:
# conf__adaptive__wIF1080h_qIF030__wAE1080h_qAE975
# conf__blend_cap020__wIF720h_qIF010__wAE1080h_qAE975
# conf__pattern_dwell3__wIF720h_qIF010__wAE1080h_qAE975
```

Where:
- `variant` matches the label source (`adaptive`, `blend_cap{ddd}`, `pattern_dwell{k}`).
- `tag` encodes the combo (`wIF{h}h_qIF{qqq}__wAE{h}h_qAE{qqq}`).

---

## Usage in scripts

- `05_compute_confidence_combo.py` → uses `percentile_rank_combo` with **fixed q**.  
- `05_compute_confidence_qt.py` → uses `percentile_rank_qt` with **threshold series** if present.  
- Both scripts:
  1. Load the latest dataset (`scores_with_blends.csv` preferred).  
  2. Find all `hybrid_label_*` columns.  
  3. For each, parse its `tag` and variant, compute IF/AE confidences with aligned windows.  
  4. Combine to a single `conf__...` column using emission policy and aggregation knobs.  
  5. Write `complete_scores_thresholds_confidence.csv`.

---

## Gotchas

- **Alignment:** use the **same `window_hours` and `min_periods`** as thresholds.  
- **Excluding current row:** we always shift by 1 in ECDF windows to avoid look‑ahead.  
- **NaNs at start:** controlled by `CONF_MIN_PERIODS` (`inherit` often equals thresholds).  
- **Label masking:** confidence is only emitted for the label classes in `CONF_EMIT_TYPES`.  
- **Clipping:** final values are clipped to `[CONF_CLIP_MIN, CONF_CLIP_MAX]` after adding `CONF_EPS`.

---

## When to use `combo` vs `qt`

- Use **`combo`** for **stable, comparable** confidence across runs and combos (simpler, faster).  
- Use **`qt`** when **blending/dwell** alters thresholds and you want confidence to track the **realised** cut‑offs (`θ_t`).

---

## Minimal example

```python
from src.confidence_scoring.percentile_rank_combo import rolling_percentile_conf

conf_if = rolling_percentile_conf(
    df["if_score"], tail="low",
    window_hours=1080, min_periods=1080,  # 45d
    threshold_q=0.03, eps=1e-9, clip_min=1e-3, clip_max=0.999
)
```

Use the same pattern for AE with `tail="high"` and the AE window/quantile.
