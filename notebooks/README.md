# COMP1885 — Static vs Adaptive Thresholds (Counts & Visual Review) (/notebooks/comapre_thresholds.ipynb)

## Overview
This notebook compares **static** thresholds with three **adaptive** variants (plain adaptive, capped–minmax, and dwell) and adds **confidence scoring** for interpretability. It consumes summary CSVs from `results/final/`, scores methods for *calm suppression* vs *event responsiveness*, shortlists the top three, and generates diagnostic figures.

## Inputs (expected under `results/final/`)
- `ALL_COMBOS_global_counts_summary.csv`  
- `ALL_COMBOS_window_counts_totals.csv`  
- `ALL_COMBOS_window_counts_detail.csv`

## What the notebook produces
- A shortlist of the top three method variants (one per family: adaptive / capped–minmax / dwell) based on weighted calm vs event scores.  
- Figures saved to `results/final/figures/` in these subfolders:  
  - `eval_windows_static_vs_top3_adaptive_variants/` – **Stacks:** raw variables with anomaly markers (size/alpha = confidence).  
  - `anomaly_density/<window>/` – **Window densities:** grouped bars of Point / Pattern / Compound.  
  - `confidence_timelines/` – **Confidence timelines** (adaptive only).  
  - `threshold_drift/` – **Model scores vs thresholds** (static lines or rolling).  
  - `anomaly_density/global_df_g/` – **Global counts & rates/1k**.

## Quick start
1. Ensure Python with **pandas**, **numpy**, **matplotlib** is available.  
2. Place the three CSVs listed above under `results/final/`.  
3. Open and run the notebook end-to-end. Output figures and audit CSVs are written under `results/final/figures/…`.

## How methods are scored & shortlisted
- Windows are tagged *Calm* or *Storm/Event* and deltas (variant − static) are computed per window.  
- A weighted score is applied (default **0.40** calm, **0.60** storm; global alignment optional) and the **best of each family** is shortlisted. Edit the `WEIGHTS` block to change emphasis.

## Figure guide (what to look for)

| Figure type | Folder | Use it to… |
|---|---|---|
| Stacks (4 raw variables + markers) | `…/eval_windows_static_vs_top3_adaptive_variants/` | Visually relate anomalies to weather signals; marker size/alpha reflect confidence. |
| Window anomaly densities | `…/anomaly_density/<window>/` | Compare Point / Pattern / Compound counts by method within each evaluation window. |
| Confidence timelines | `…/confidence_timelines/` | Check whether higher confidence aligns with denser anomaly clusters. |
| Threshold drift (IF & AE) | `…/threshold_drift/` | Inspect how rolling thresholds track scores vs static lines; exceedance dots show flags. |

## Key rules & assumptions
- **Labels:** adaptive/hybrid labels come from trusted `hybrid_label_*` columns; **static** hybrid is rebuilt from `is_if_anomaly` & `is_lstm_anomaly`.  
- **Confidence:** percentile-rank per rolling window, mapped to **marker size** and **alpha** (0–1).  
- **Windows:** evaluation windows are **inclusive** of start and end timestamps.

## Customisation tips
- **Change calm vs event emphasis:** edit the `WEIGHTS` block in the “Score Methods” section.  
- **Toggle filters:** options exist for global % deviation bounds or enforcing non-decrease in storms.  
- **Add/replace evaluation windows:** update the `EVAL_WINDOWS` list.

## Upstream context
The upstream pipeline generates the inputs and, if required, confidence and threshold sidecars for the shortlisted methods. This notebook assumes those files are present and focuses on **comparison, shortlisting, and visual diagnostics**.