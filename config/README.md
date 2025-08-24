# Config Directory

This folder contains all **project settings**.  
It controls how the pipeline runs without changing the code.

---

## Files

### `.env` / `.env.example`
Environment file with all key parameters:

- **Data paths** – input data and trained models.  
- **Static thresholds** – fixed cut-offs from COMP1884 (used as baselines).  
- **Adaptive thresholds** – rolling window sizes (`14d`, `30d`, `45h`) and quantiles.  
- **Combo scope** – choose between all window/quantile combinations or only those listed.  
- **Evaluation windows** – named date ranges (e.g. calm vs storm) for comparing results.  
- **Blending settings** – rules like *capped min–max* and *dwell pattern*.  
- **Confidence scoring** – how confidence values are built (method, tails, emission policy).  

### `config.py`
Python module that:

- Reads `.env` and validates settings.  
- Parses tokens like `14d` → hours.  
- Builds the **full grid of combos** or applies only the selected ones.  
- Prepares evaluation windows, blend specs, and confidence knobs.  
- Creates a unique **run slug** so every run has its own results folder.  
- Writes a `README.txt` into each run folder with the parameters used.  

---

## How It Works

1. You edit `.env` to change experiment settings.  
2. `config.py` loads these into Python objects.  
3. A new **slugged run folder** is created in `results/interim/`.  
4. A summary `README.txt` is written for reproducibility.  

---

## Key Idea

- **All behaviour is driven by `.env`.**  
- **Scripts never need manual edits.**  
- Every run is traceable, with its own parameters and outputs.
    