# src/utils

This folder provides helper functions for data handling, run context, and naming.  
These utilities keep the main pipeline scripts clean and consistent.

---

## Files

### `__init__.py`
- Exports the main helpers so they can be imported directly.
- Groups:
  - **Data helpers** (from `data_utils.py`)
  - **Run context helpers** (from `run_context.py`)

---

### `data_utils.py`
Helpers for working with the dataset and DataFrames.

- **load_scores(path)** → Load scores CSV, ensure timestamps sorted.
- **slice_by_date(df, col, start, end)** → Inclusive slice by date range.
- **mask_valid_rows(df, cols)** → Mask rows with no missing values in given cols.
- **require_columns(df, cols, context)** → Raise clear error if missing cols.
- **ensure_nullable_bools(df, cols)** → Convert cols to pandas nullable boolean dtype.
- **write_csv(df, path, log_prefix)** → Save DataFrame to CSV with log message.
- **ensure_static_flags(...)** → Create static anomaly flags (IF ≤ threshold, AE ≥ threshold).

---

### `run_context.py`
Helpers for run management and reproducibility.

- **compute_slug(fingerprint)** → Make unique ID (hash) from config knobs.
- **results_dir(base_dir, slug)** → Build results directory path.
- **write_latest_pointer(root, slug)** → Save LATEST_RUN.txt pointer to latest run.
- **write_readme(run_dir, slug, summary)** → Save README.txt with run summary.
- **_format_readme(...)** → Format run README text (slug + key params).

---

### `naming.py`
Canonical naming rules for thresholds, flags, hybrids, and confidence columns.

- **tag(w_if_h, q_if, w_ae_h, q_ae)** → Build canonical combo tag.
- **thr_if_adapt / flg_if_adapt / thr_ae_adapt / flg_ae_adapt** → Adaptive col names.
- **thr_if_blend / flg_if_blend** → Blended thresholds and flags.
- **dwell(colname, k)** → Add dwell suffix.
- **hybrid_name(kind, combo_tag)** → Hybrid label col name.
- **conf_if / conf_ae / conf_final** → Confidence column names.

---

## Purpose
- Keep code DRY and readable.
- Enforce consistent naming and logging.
- Ensure reproducibility of runs with slugs + READMEs.
