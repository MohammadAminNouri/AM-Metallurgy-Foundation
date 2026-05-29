import numpy as np
import pandas as pd


def add_energy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add AM process physics-aware energy descriptors.

    Uses W, mm/s, um -> converts um to mm where needed.
    """
    out = df.copy()
    p = out.get("laser_power_W")
    v = out.get("scan_speed_mm_s")
    h_um = out.get("hatch_spacing_um")
    t_um = out.get("layer_thickness_um")
    b_um = out.get("beam_diameter_um")

    if p is not None and v is not None:
        out["linear_energy_density_J_mm"] = p / v.replace(0, np.nan)
        out["power_speed_ratio"] = out["linear_energy_density_J_mm"]
    if p is not None and v is not None and h_um is not None and t_um is not None:
        h_mm = h_um / 1000
        t_mm = t_um / 1000
        out["volumetric_energy_density_J_mm3"] = p / (v.replace(0, np.nan) * h_mm.replace(0, np.nan) * t_mm.replace(0, np.nan))
    if p is not None and b_um is not None:
        b_mm = b_um / 1000
        area = np.pi * (b_mm / 2) ** 2
        out["beam_power_density_W_mm2"] = p / area.replace(0, np.nan)
    return out
