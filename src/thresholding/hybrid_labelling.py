# src/thresholding/hybrid_labelling.py

import pandas as pd

def hybrid_from_flags(if_flag: pd.Series, ae_flag: pd.Series) -> pd.Series:
    lab = pd.Series(pd.NA, index=if_flag.index, dtype="string")
    both = (if_flag == True) & (ae_flag == True)
    only_if = (if_flag == True) & (ae_flag != True)
    only_ae = (ae_flag == True) & (if_flag != True)
    return lab.mask(only_if, "Point").mask(only_ae, "Pattern").mask(both, "Compound")