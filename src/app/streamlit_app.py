import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd
import streamlit as st

from src.features.build_features import build_all_features
from src.models.train_baselines import benchmark_models, train_best_model
from src.trust.out_of_domain_check import domain_report
from src.trust.nearest_literature_cases import nearest_cases

st.set_page_config(page_title="OpenMetalAM-AI", layout="wide")
st.title("OpenMetalAM-AI")
st.caption("Literature-mined, metallurgy-aware ML for metal additive manufacturing")

TARGETS = [
    "yield_strength_MPa", "UTS_MPa", "elongation_percent", "hardness_HV",
    "elastic_modulus_GPa", "surface_roughness_Ra_um", "relative_density_percent"
]

DEFAULT_FEATURES = [
    "alloy", "AM_process", "AM_subprocess", "machine", "laser_power_W", "scan_speed_mm_s",
    "hatch_spacing_um", "layer_thickness_um", "beam_diameter_um", "build_orientation",
    "specimen_orientation", "heat_treatment", "relative_density_percent", "porosity_percent",
    "grain_size_um", "grain_morphology", "texture_intensity_MRD", "matrix_phase", "secondary_phases",
    "linear_energy_density_J_mm", "volumetric_energy_density_J_mm3", "beam_power_density_W_mm2"
]

page = st.sidebar.radio("Page", [
    "1 Dataset explorer", "2 Train benchmark", "3 Predict user input", "4 Data requirements"
])

uploaded_train = st.sidebar.file_uploader("Upload training dataset CSV", type=["csv"], key="train")
if uploaded_train:
    raw_train = pd.read_csv(uploaded_train)
else:
    example_path = ROOT / "data" / "examples" / "user_input_template.csv"
    raw_train = pd.read_csv(example_path)

train_df = build_all_features(raw_train)

if page == "1 Dataset explorer":
    st.subheader("Dataset")
    st.write(f"Rows: {len(train_df)} | Columns: {len(train_df.columns)}")
    st.dataframe(train_df, use_container_width=True)
    st.subheader("Missing values")
    miss = train_df.isna().mean().sort_values(ascending=False).rename("missing_fraction")
    st.dataframe(miss.to_frame(), use_container_width=True)

elif page == "2 Train benchmark":
    st.subheader("Train benchmark models")
    available_targets = [t for t in TARGETS if t in train_df.columns and train_df[t].notna().sum() >= 2]
    if not available_targets:
        st.warning("Upload a dataset with at least one target column and at least two non-missing labels.")
    else:
        target = st.selectbox("Target", available_targets)
        feature_cols = [c for c in DEFAULT_FEATURES if c in train_df.columns and c != target]
        st.write("Using features:", feature_cols)
        if st.button("Run benchmark"):
            try:
                result = benchmark_models(train_df, target, feature_cols, n_splits=5)
                st.dataframe(result, use_container_width=True)
            except Exception as e:
                st.error(f"Benchmark failed: {e}")

elif page == "3 Predict user input":
    st.subheader("Predict with user input")
    available_targets = [t for t in TARGETS if t in train_df.columns and train_df[t].notna().sum() >= 2]
    target = st.selectbox("Target to train", available_targets) if available_targets else None
    uploaded_query = st.file_uploader("Upload query/input CSV", type=["csv"], key="query")
    if uploaded_query:
        query_raw = pd.read_csv(uploaded_query)
    else:
        query_raw = raw_train.head(1).drop(columns=[c for c in TARGETS if c in raw_train.columns], errors="ignore")
    query_df = build_all_features(query_raw)
    st.write("Input rows")
    st.dataframe(query_df, use_container_width=True)
    if target and st.button("Train and predict"):
        feature_cols = [c for c in DEFAULT_FEATURES if c in train_df.columns and c in query_df.columns and c != target]
        try:
            model = train_best_model(train_df, target, feature_cols, model_name="random_forest")
            preds = model.predict(query_df[feature_cols])
            out = query_df.copy()
            out[f"predicted_{target}"] = preds
            st.subheader("Predictions")
            st.dataframe(out, use_container_width=True)
            warnings = domain_report(train_df, query_df)
            if warnings:
                st.subheader("Warnings")
                for w in warnings:
                    st.warning(w)
            st.subheader("Nearest literature/training cases")
            st.dataframe(nearest_cases(train_df.dropna(subset=[target]), query_df, feature_cols, k=min(5, len(train_df))), use_container_width=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

else:
    st.subheader("Data requirements")
    st.markdown("""
    Minimal columns for process-property prediction:
    - alloy
    - AM_process
    - laser_power_W
    - scan_speed_mm_s
    - layer_thickness_um
    - one target such as yield_strength_MPa or UTS_MPa

    Stronger columns:
    - hatch_spacing_um
    - heat_treatment
    - specimen_orientation
    - relative_density_percent
    - porosity_percent
    - grain_size_um
    - grain_morphology
    - matrix_phase
    - secondary_phases
    - texture_intensity_MRD
    """)
