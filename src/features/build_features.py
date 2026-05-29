from .energy_density import add_energy_features
from .thermal_features import add_thermal_features
from .composition_features import add_simple_composition_features


def build_all_features(df):
    out = add_energy_features(df)
    out = add_thermal_features(out)
    out = add_simple_composition_features(out)
    return out
