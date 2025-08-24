# Scripts

These scripts run the whole pipeline end-to-end or step-by-step.  
Language is simple; filenames match exactly.

---

## Quick start

1. Set your settings in `config/.env`.
2. (Optional) create the conda env:  
   ```bash
   conda env create -f environment.yml
   conda activate comp1885
   ```
3. Run everything:  
   ```bash
   python scripts/run_full_pipeline.py
   ```
4. Outputs go to `results/interim/<slug>/`.  
   A `README.txt` is written there with the exact parameters.

---

## Run order (what happens)

1. **01_generate_thresholds.py**  
   Builds *adaptive* thresholds and flags for each combo; writes a *sidecar* map.
2. **02_generate_blends.py**  
   Applies **CAPPED_minmax** and/or **DWELL_PATTERN**; writes blended flags + labels.
3. **03_generate_counts.py**  
   Counts Point/Pattern/Compound per *evaluation window*.
4. **04_generate_counts_global.py**  
   Global totals vs static baseline (whole dataset).
5. **05_compute_confidence_[combo|qt].py**  
   Adds confidence columns aligned to the final labels from (2).

The runner picks `05_compute_confidence_combo.py` or `05_compute_confidence_qt.py`
based on `CONF_IMPL` in `.env`.

---

## Script details

### 01_generate_thresholds.py
**Purpose:** compute rolling quantile thresholds and flags per combo; make adaptive hybrid labels.

**Inputs**
- Data from `config.DATA_PATH` (`timestamp`, `if_score`, `lstm_score`).
- Settings from `.env`: `WINDOW_*`, `PERCENTILES_*`, `COMBOS_SCOPE`, `SELECTED_COMBOS`,
  `ADAPTIVE_MIN_PERIODS`, static thresholds and static flag column names.

**Writes (to `results/interim/<slug>/`)**
- `scores_with_thresholds.csv` – original data +:
  - Adaptive thresholds, flags, hybrid labels
  - Static flags if missing
- `columns_map.json` – sidecar with combo metadata and column names.

---

### 02_generate_blends.py
**Purpose:** apply blending to thresholds/flags and produce final hybrid labels.

**Inputs**
- Outputs from step 01.
- `.env` blend specs.

**Writes**
- `scores_with_blends.csv` – adds blended thresholds, flags, dwell flags, and new hybrid labels.
- Updates `columns_map.json`.

**Logic**
- **CAPPED_minmax:** limits adaptive thresholds within a cap relative to static thresholds.  
- **DWELL_PATTERN:** requires k consecutive AE anomalies before flagging (IF unchanged).

---

### 03_generate_counts.py
**Purpose:** per-window counts of anomaly types from the **latest available** labels.

**Inputs**
- Prefers `scores_with_blends.csv`, else `scores_with_thresholds.csv`.
- `columns_map.json`.
- `.env` `EVAL_WINDOWS`.

**Writes**
- `window_counts_detail.csv` – per window × combo.  
- `window_counts_totals.csv` – totals per window.

---

### 04_generate_counts_global.py
**Purpose:** global totals vs static baseline over the entire dataset.

**Inputs**
- Prefers `scores_with_blends.csv`, else `scores_with_thresholds.csv`.
- `columns_map.json`.

**Writes**
- `global_counts_summary.csv` – adaptive vs static counts, with diffs and percentages.

---

### 05_compute_confidence_combo.py
**Purpose:** confidence from **percentile rank vs fixed combo quantiles**.

**When used:** `CONF_IMPL=combo`.

**Writes**
- `complete_scores_thresholds_confidence.csv` – adds confidence columns.

---

### 05_compute_confidence_qt.py
**Purpose:** confidence from **percentile rank vs realised thresholds** (time-varying).

**When used:** `CONF_IMPL=qt`.

**Writes**
- Same as above.

---

## Column naming (quick reference)

- **Combo tag:** `wIF{h}h_qIF{qqq}__wAE{h}h_qAE{qqq}`  
- **Hybrid labels:** `hybrid_label_{variant}__{tag}`  
- **Blended thresholds/flags:** `*_blend_cap{ddd}`, `*_blend_flag_cap{ddd}`  
- **Dwell flags:** `*_dwell{k}`  
- **Confidence:** `conf__{variant}__{tag}`

---

## Common tips

- Run step 01 first if inputs are missing.  
- If `COMBOS_SCOPE=selected`, set `SELECTED_COMBOS`.  
- Use `ADAPTIVE_MIN_PERIODS=1` to avoid initial NAs.  
- Changing `.env` changes the run **slug**; outputs are always separate.  
- You can delete old folders under `results/interim/` safely.

---

## Single-step runs (optional)

Run any script directly:
```bash
python scripts/01_generate_thresholds.py
python scripts/02_generate_blends.py
python scripts/03_generate_counts.py
python scripts/04_generate_counts_global.py
python scripts/05_compute_confidence_combo.py
python scripts/05_compute_confidence_qt.py
```
Make sure earlier steps have produced their inputs first.
