import pandas as pd
from src.utils import mask_valid_rows
from src.thresholding.hybrid_labelling import hybrid_from_flags

ANOM_ORDER = ["Point", "Pattern", "Compound"]

def count_from_label_series(label: pd.Series) -> pd.DataFrame:
    """
    Return a tidy count table in fixed ANOM_ORDER with zeros for missing classes.
    """
    vc = label.value_counts(dropna=True)
    out = (
        pd.DataFrame({"anomaly_type": ANOM_ORDER})
        .merge(vc.rename("count").rename_axis("anomaly_type").reset_index(),
               on="anomaly_type", how="left")
        .fillna({"count": 0})
        .astype({"count": int})
    )
    return out

def counts_variant_vs_static(
    dfw: pd.DataFrame,
    if_col: str,
    ae_col: str,
    static_if_col: str,
    static_ae_col: str,
) -> pd.DataFrame:
    """
    Compare counts from a 'variant' pair of flags (if_col, ae_col) to the static baseline
    (static_if_col, static_ae_col) on the same validity mask.

    Returns
    -------
    DataFrame with columns:
      ["anomaly_type", "count_variant", "count_static", "valid_n"]
    """
    m = mask_valid_rows(dfw, [if_col, ae_col, static_if_col, static_ae_col])
    valid_n = int(m.sum())

    # Variant
    variant_labels = hybrid_from_flags(
        dfw.loc[m, if_col].astype("boolean"),
        dfw.loc[m, ae_col].astype("boolean"),
    )
    a = count_from_label_series(variant_labels).rename(columns={"count": "count_variant"})

    # Static
    static_labels = hybrid_from_flags(
        dfw.loc[m, static_if_col].astype("boolean"),
        dfw.loc[m, static_ae_col].astype("boolean"),
    )
    s = count_from_label_series(static_labels).rename(columns={"count": "count_static"})

    out = a.merge(s, on="anomaly_type", how="left")
    out["valid_n"] = valid_n
    return out[["anomaly_type", "count_variant", "count_static", "valid_n"]]
