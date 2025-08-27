# COMP1885 Adaptive Thresholding Pipeline

This project extends the COMP1884 anomaly detection work with **adaptive thresholds**, **threshold blending (capped min–max + dwell)**, and **rolling confidence scoring** for a hybrid **Isolation Forest (IF)** + **LSTM Autoencoder (AE)** system. It also includes a companion analysis notebook for ranking threshold combinations and generating figures.

---

## Repository Structure

```
.
├─ config/                  # .env + config loader (config.py)
├─ data/                    # Input scores CSV from COMP1884
├─ models/                  # Trained IF + LSTM-AE (not tracked in Git)
├─ notebooks/               # Analysis notebooks (e.g., compare_thresholds.ipynb)
├─ results/
│  ├─ interim/<slug>/       # One folder per run (full artefacts; not tracked)
│  └─ final/                # Curated outputs for reporting (tracked)
├─ scripts/                 # 01–05 pipeline steps + runner
└─ src/
   ├─ thresholding/         # Static/adaptive thresholds, blends, hybrid labels, counts
   ├─ confidence_scoring/   # Rolling percentile-based confidence
   └─ utils/                # Data helpers, naming, run context
```

> **Note:** `results/final/figures/` is where the notebook writes publication-ready plots:
> - `eval_windows_static_vs_top3_adaptive_variants/` (4× stacked time-series per method)
> - `anomaly_density/` (per-window counts by type & method)
> - `confidence_timelines/`
> - `threshold_drift/`
> - `anomaly_density/global_df_g/` (global density from `df_g` summary)

---

## Quick Start

1. **Environment**
   ```bash
   conda env create -f environment.yml && conda activate comp1885
   # or
   pip install -r requirements.txt
   ```

2. **Configure** `config/.env` (see `.env.example`). Key knobs are documented below.

3. **Run the full pipeline**
   ```bash
   python scripts/run_full_pipeline.py
   ```
   Outputs go to `results/interim/<slug>/` (full artefacts) and selected items are copied to `results/final/`.

4. **Open the analysis notebook** `notebooks/compare_thresholds.ipynb` to:
   - rank IF/AE window–quantile combinations,
   - shortlist top variants (weighted Calm vs Event performance),
   - produce figures into `results/final/figures/...`.

---

## Pipeline Steps

1. **01_generate_thresholds.py** – compute static + adaptive rolling quantiles, generate model flags and **hybrid labels** (Point/Pattern/Compound).  
   _Outputs:_ `scores_with_thresholds.csv`, `columns_map.json`.

2. **02_generate_blends.py** – apply **capped min–max** (cap ±δ) and **dwell (k)** to thresholds/flags.  
   _Outputs:_ `scores_with_blends.csv` (+ sidecar updates).

3. **03_generate_counts.py** – per-window counts by anomaly type and method.  
   _Outputs:_ `window_counts_detail.csv`, `window_counts_totals.csv`.

4. **04_generate_counts_global.py** – global totals vs static baseline.  
   _Outputs:_ `global_counts_summary.csv`.

5. **05_compute_confidence_[combo|qt].py** – rolling percentile-rank **confidence** in [0,1].  
   _Outputs:_ `complete_scores_thresholds_confidence.csv` (+ final shortlist CSV when scope=selected).

`run_full_pipeline.py` executes steps 01–05 in order, using `config/.env` to control **scope**, **blends**, **windows**, and **confidence**.

---

## Configuration (config/.env)

- **Static thresholds** (from COMP1884):
  ```ini
  STATIC_THRESH_IF=0.03304385848702787
  STATIC_THRESH_AE=0.6425
  STATIC_THRESH_QUANT_IF=0.03
  STATIC_THRESH_QUANT_AE=0.95
  ```

- **Adaptive combos** (IF/AE windows & quantiles):
  ```ini
  WINDOW_IF_LIST=14d,30d,45d,60d
  WINDOW_AE_LIST=14d,30d,45d
  PERCENTILES_IF=0.01,0.03
  PERCENTILES_AE=0.95,0.975,0.99
  COMBO_WINDOW_UNITS=days
  ADAPTIVE_MIN_PERIODS=1      # '1' = warm from row 1; 'window' = require full window
  ```

- **Scope** (all combinations vs hand-picked):
  ```ini
  COMBOS_SCOPE=all | selected
  SELECTED_COMBOS=45d:0.01:30d:0.95,14d:0.03:14d:0.975,14d:0.03:14d:0.95
  ```

- **Blends**:
  ```ini
  BLENDS=CAPPED_minmax,DWELL_PATTERN

  # Capped min–max (cap δ applied symmetrically around the adaptive threshold)
  BLEND_CAPPED_minmax=use:all;op:capped_minmax;cap:0.02

  # Dwell on AE flags (require k consecutive AE=1); IF is unchanged
  BLEND_DWELL_PATTERN=use:all;op:dwell_pattern;k:3
  ```

- **Confidence** (rolling percentile rank):
  ```ini
  CONF_ENABLED=1
  CONF_METHOD=percentile_rank
  CONF_TAIL_IF=low     # smaller IF score = more anomalous
  CONF_TAIL_AE=high    # larger AE error = more anomalous
  CONF_IMPL=qt         # use combo quantiles (qt) for rank windows
  CONF_EMIT_TYPES=Point,Pattern,Compound
  CONF_CLIP_MIN=0.001
  CONF_CLIP_MAX=0.999
  ```

- **Evaluation windows** (inclusive):
  ```ini
  EVAL_WINDOWS=Calm_2019|2019-05-12|2019-05-24,...
  ```

Each run writes a `README.txt` with resolved parameters to `results/interim/<slug>/` and a `LATEST_RUN.txt` pointer.

---

## Analysis Notebook (compare_thresholds.ipynb)

The notebook supports **two passes**:

1. **Explore all combinations (no confidence):**
   - Set `COMBOS_SCOPE=all` and `CONF_ENABLED=0`.
   - Run the pipeline to produce:
     - `ALL_COMBOS_global_counts_summary.csv`
     - `ALL_COMBOS_window_counts_detail.csv`
     - `ALL_COMBOS_window_counts_totals.csv`
   - In the notebook, load these CSVs, label windows (Calm/Event), and compute a weighted score (default **0.4 Calm suppression**, **0.6 Event responsiveness**). Review scatter plots and the ranking table.

2. **Focus on top-3 (with confidence & figures):**
   - Set `COMBOS_SCOPE=selected` and put the **three winners** into `SELECTED_COMBOS`.
   - Enable confidence: `CONF_ENABLED=1`.
   - Re-run the pipeline to produce `SHORTLIST_complete_scores_thresholds_confidence.csv`.
   - In the notebook, run the **param cell** once (common helpers) then the **plotting cells**:
     - **Stacks**: 4× time series + coloured anomaly markers sized/alpha by confidence.
     - **Anomaly density**: grouped bars per window (Point/Pattern/Compound by method).
     - **Confidence timelines**: confidence vs time with anomaly overlays.
     - **Threshold drift**: IF/AE scores vs (static/adaptive/blended) thresholds.

All figures are saved under `results/final/figures/...` with consistent colours:
- **Point (IF)** = blue `#1f77b4`
- **Pattern (AE)** = orange `#ff7f0e`
- **Compound (IF+AE)** = purple `#9467bd`

---

## Methods (in brief)

- **Adaptive thresholds** – rolling quantiles for IF (lower is worse) and AE (higher is worse) with aligned windows.
- **Capped min–max** – clamp each adaptive threshold to a **±cap** band around the **static** threshold to prevent drift that is *too* aggressive or *too* timid.
- **Dwell (k)** – applied **to AE flags only**; an AE anomaly must persist for at least **k** consecutive timestamps to be counted. This suppresses short, noisy runs in calm periods.
- **Hybrid label** – combine IF and AE flags at each timestamp to one of: **Point** (IF only), **Pattern** (AE only), **Compound** (both), or **None**.
- **Confidence** – rolling percentile rank on the combo’s window: IF uses the **lower** tail, AE uses the **upper** tail; **Compound** aggregates (mean) of IF/AE confidences.

---

## Figure Index (what to use where)

- **Stacks** (`figures/eval_windows_static_vs_top3_adaptive_variants/`)  
  Use to show how each method behaves on **Calm** vs **Event** windows, with anomaly markers sized by confidence.

- **Anomaly density** (`figures/anomaly_density/<Window>/`)  
  Use to compare **counts** of Point/Pattern/Compound across methods within each evaluation window.

- **Confidence timelines** (`figures/confidence_timelines/`)  
  Use when discussing how confidence tracks the signal and correlates with emitted anomalies.

- **Threshold drift** (`figures/threshold_drift/`)  
  Use to inspect how **scores interact with thresholds** (static vs adaptive vs blended), and where exceedance flags occur.

- **Global density (df_g)** (`figures/anomaly_density/global_df_g/`)  
  Use for **overall** count/rate comparison across the entire dataset.

---

## Reproducing the Shortlist

1. Run **all combinations** (no confidence).  
2. In the notebook, set weights and compute **`composite_score`**; take the **top-3** methods (one per family: adaptive, capped, dwell).  
3. Switch to **selected** scope with these three in `.env` → rerun (with confidence).  
4. Generate **figures** and **window tables** from the notebook.

The shortlist used in the report:
- `adaptive_IFw1080q010_AEw720q950`
- `CAPPED_minmax_IFw336q030_AEw336q975` (cap=0.02)
- `DWELL_PATTERN_IFw336q030_AEw336q950` (k=3)

---

## Practical Tuning Tips

- **Increase cap (capped min–max)** → lets adaptive thresholds drift **further** from static; can **raise** AE sensitivity to large errors and **lower** IF sensitivity to small scores, depending on the relative positions.  
- **Decrease cap** → keeps thresholds closer to static, often **reducing** calm-period noise but may **miss** emerging events.
- **Increase dwell k** → requires longer AE runs, **suppressing** isolated/short patterns in calm windows.  
- **Decrease dwell k** → **more** AE anomalies admitted (more sensitive to short bursts).
- Use **window sizes** to match signal volatility: **longer** windows yield steadier thresholds; **shorter** windows react faster.

Always validate the choices with the **window density plots** and **threshold drift** figures.

---

## Troubleshooting

- **“Missing hybrid col … skipping”** – your data slice lacks the expected hybrid/threshold/confidence column (likely due to scope or a previous run’s settings). Re-run with the correct `COMBOS_SCOPE`, `BLENDS`, and confidence settings, or trim the plotting code to available columns.
- **No anomalies in a Calm window** – expected if thresholds are conservative (e.g., high AE threshold after blend + dwell). Verify with the **threshold drift** figure.
- **Confidence not plotted** – only methods with `conf__...` columns emit confidence; static has none by design.

---

## References

- **COMP1884** – baseline IF & LSTM-AE models + static thresholds.  
- **COMP1885** – adaptive thresholds, capped min–max, dwell, and confidence scoring; ranking & visual diagnostics via notebook.
