import numpy as np
import pandas as pd


def add_thermal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic material/thermal features when columns exist."""
    out = df.copy()
    k = out.get("thermal_conductivity_W_mK")
    rho = out.get("density_kg_m3")
    cp = out.get("specific_heat_J_kgK")
    if k is not None and rho is not None and cp is not None:
        out["thermal_diffusivity_m2_s"] = k / (rho.replace(0, np.nan) * cp.replace(0, np.nan))
    if "melting_temperature_C" in out and "preheat_temperature_C" in out:
        out["thermal_headroom_C"] = out["melting_temperature_C"] - out["preheat_temperature_C"]
    return out
