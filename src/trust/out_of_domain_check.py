import pandas as pd

NUMERIC_DOMAIN_COLUMNS = [
    "laser_power_W", "scan_speed_mm_s", "hatch_spacing_um", "layer_thickness_um",
    "volumetric_energy_density_J_mm3", "linear_energy_density_J_mm"
]


def domain_report(train_df: pd.DataFrame, user_df: pd.DataFrame) -> list[str]:
    """Return human-readable out-of-domain warnings."""
    warnings = []
    for col in NUMERIC_DOMAIN_COLUMNS:
        if col in train_df.columns and col in user_df.columns:
            train_vals = train_df[col].dropna()
            if train_vals.empty:
                continue
            lo, hi = train_vals.min(), train_vals.max()
            for idx, val in user_df[col].dropna().items():
                if val < lo or val > hi:
                    warnings.append(f"Row {idx}: {col}={val} is outside training range [{lo}, {hi}].")
    for col in ["alloy", "AM_process", "AM_subprocess", "heat_treatment"]:
        if col in train_df.columns and col in user_df.columns:
            known = set(train_df[col].dropna().astype(str).str.lower())
            for idx, val in user_df[col].dropna().astype(str).items():
                if val.lower() not in known:
                    warnings.append(f"Row {idx}: {col}='{val}' was not seen in training data.")
    return warnings
