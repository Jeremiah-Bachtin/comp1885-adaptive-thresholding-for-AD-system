# src/thresholding/ids.py
def q_to_str(q: float) -> str:
    """
    Convert a quantile (e.g., 0.975, 0.030, 0.010) to a 3-digit string:
      0.975 -> '975', 0.030 -> '030', 0.010 -> '010'
    """
    return str(int(round(q * 1000))).zfill(3)

def combo_id(w_if: int, q_if: float, w_ae: int, q_ae: float) -> str:
    """
    Build the canonical combo key used in sidecar and scripts.

    Notes
    -----
    - w_if and w_ae are window sizes in HOURS (not days).
      For example: 336 = 14 days, 720 = 30 days.
    - q_if and q_ae are quantiles (floats between 0–1).
    - The combo ID string encodes these as:
        IFw{w_if}q{q_if*1000:03d}_AEw{w_ae}q{q_ae*1000:03d}

    Example
    -------
    >> combo_id(45, 0.03, 45, 0.975)
    'IFw45q030_AEw45q975'
    """
    return f"IFw{int(w_if)}q{q_to_str(q_if)}_AEw{int(w_ae)}q{q_to_str(q_ae)}"
