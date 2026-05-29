import pandas as pd


def convert_common_units(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common AM units in-place-safe style."""
    out = df.copy()
    if "hatch_spacing_mm" in out and "hatch_spacing_um" not in out:
        out["hatch_spacing_um"] = out["hatch_spacing_mm"] * 1000
    if "layer_thickness_mm" in out and "layer_thickness_um" not in out:
        out["layer_thickness_um"] = out["layer_thickness_mm"] * 1000
    if "scan_speed_mm_min" in out and "scan_speed_mm_s" not in out:
        out["scan_speed_mm_s"] = out["scan_speed_mm_min"] / 60
    if "yield_strength_GPa" in out and "yield_strength_MPa" not in out:
        out["yield_strength_MPa"] = out["yield_strength_GPa"] * 1000
    return out
