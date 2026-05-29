import pandas as pd

ELEMENT_COLUMNS = [
    "Fe", "Cr", "Ni", "Mo", "Al", "Mn", "Si", "Mg", "Nb", "Ti", "V", "Co", "Zr", "Cu", "W", "N", "C", "Sn", "Zn", "Sc"
]


def add_simple_composition_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    present = [c for c in ELEMENT_COLUMNS if c in out.columns]
    if present:
        out["n_alloying_elements_reported"] = (out[present].fillna(0) > 0).sum(axis=1)
        out["base_element"] = out[present].fillna(0).idxmax(axis=1)
    return out
