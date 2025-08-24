# COMP1885 Adaptive Thresholding Pipeline

This project builds on the COMP1884 anomaly detection work.  
It adds **adaptive thresholds, blending, and confidence scoring** for hybrid IF + LSTM-AE models.

---

## Structure

```
.
├─ config/              # Environment (.env) + config loader (config.py)
├─ data/                # Input scores CSV (from COMP1884)
├─ models/              # Trained IF + LSTM-AE models (reused from COMP1884)
├─ notebooks/           # Exploratory and dev notebooks (WIP)
├─ results/             # All outputs (final + interim run folders)
├─ scripts/             # Pipeline steps (01–05 + runner)
└─ src/                 # Core library code
   ├─ thresholding/     # Thresholds, blending, hybrid labels, counts
   ├─ confidence_scoring/ # Rolling percentile-based confidence scores
   └─ utils/            # Data helpers, naming, run context
```

---

## Quick start

1. Clone the repo and create the environment:
   ```bash
   conda env create -f environment.yml
   conda activate comp1885
   ```
   or use `pip install -r requirements.txt`.

2. Set up `config/.env` (see `.env.example` for template).

3. Run the full pipeline:
   ```bash
   python scripts/run_full_pipeline.py
   ```

4. Outputs are written to:
   - `results/interim/<slug>/` (all intermediate results, one folder per run)
   - `results/final/` (curated artefacts for reporting)

---

## Pipeline steps

1. **01_generate_thresholds.py**  
   Rolling quantile thresholds + adaptive flags + hybrid labels.  
   Writes `scores_with_thresholds.csv` + `columns_map.json`.

2. **02_generate_blends.py**  
   Applies blending (CAPPED_minmax) and dwell.  
   Writes `scores_with_blends.csv`, updates sidecar.

3. **03_generate_counts.py**  
   Per-window anomaly counts (Point / Pattern / Compound).  
   Writes `window_counts_detail.csv` + `window_counts_totals.csv`.

4. **04_generate_counts_global.py**  
   Global totals vs static baseline.  
   Writes `global_counts_summary.csv`.

5. **05_compute_confidence_[combo|qt].py**  
   Confidence scoring (percentile-rank) using fixed combo q or realised thresholds.  
   Writes `complete_scores_thresholds_confidence.csv`.

The runner (`run_full_pipeline.py`) executes steps 01–05 in order.

---

## Config-driven

- **All parameters live in `config/.env`.**  
- `config/config.py` parses them and builds a **slug** (unique run ID).  
- Each run has its own folder in `results/interim/` with a `README.txt` summarising settings.

Key groups of knobs:
- **Static thresholds** (from COMP1884 baseline).  
- **Adaptive windows & quantiles** (for IF + AE).  
- **Blending settings** (cap size, dwell k).  
- **Evaluation windows** (date ranges).  
- **Confidence settings** (method, tails, emission types).  

---

## Naming conventions

- **Adaptive thresholds:**  
  - `if_adaptive_thresh__w{h}_q{qqq}__{tag}`  
  - `lstm_adaptive_thresh__w{h}_q{qqq}__{tag}`

- **Flags:**  
  - `is_if_adaptive__...`, `is_lstm_adaptive__...`  
  - blended → `_blend_flag_cap{ddd}`  
  - dwell → `_dwell{k}`

- **Hybrid labels:**  
  - `hybrid_label_{variant}__{tag}`

- **Confidence:**  
  - `conf__{variant}__{tag}`

---

## Results folders

- **interim/**  
  - One subfolder per run (`<slug>`).  
  - Contains all intermediate outputs + `README.txt`.  
  - `LATEST_RUN.txt` points to the newest run.

- **final/**  
  - Only clean artefacts for reporting.  
  - This folder is tracked in Git.

---

## Notes

- Models are not tracked in Git (`models/` is ignored).  
- Only `results/final/` is versioned; `results/interim/` can be deleted safely.  
- All columns are timestamp-aligned and use hourly resolution.  
- Confidence values are always clipped into `[0.001, 0.999]`.

---

## References

- **COMP1884** group project → baseline models and static thresholds.  
- **COMP1885** individual project → extension with adaptive methods and confidence scoring.
