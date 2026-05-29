import pandas as pd
from src.features.build_features import build_all_features


def test_energy_features_added():
    df = pd.DataFrame({
        "laser_power_W": [200],
        "scan_speed_mm_s": [800],
        "hatch_spacing_um": [100],
        "layer_thickness_um": [30],
        "beam_diameter_um": [80],
    })
    out = build_all_features(df)
    assert "linear_energy_density_J_mm" in out.columns
    assert "volumetric_energy_density_J_mm3" in out.columns
    assert out.loc[0, "linear_energy_density_J_mm"] == 0.25
