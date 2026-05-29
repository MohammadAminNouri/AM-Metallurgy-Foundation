from src.features.build_features import build_all_features


def build_psp_dataset(process_df, micro_df=None, cryst_df=None, phase_df=None):
    """Merge process/property, microstructure, crystallography, and phase tables."""
    df = process_df.copy()
    for other in [micro_df, cryst_df, phase_df]:
        if other is not None and "sample_id" in other.columns:
            df = df.merge(other, on="sample_id", how="left", suffixes=("", "_extra"))
    return build_all_features(df)
