from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make imports work locally and on Streamlit Cloud, regardless of run directory.
APP_FILE = Path(__file__).resolve()
REPO_ROOT = APP_FILE.parents[2]  # OpenMetalAM-AI/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.build_features import build_all_features
from src.models.train_baselines import benchmark_models, train_best_model
from src.trust.out_of_domain_check import domain_report
from src.trust.nearest_literature_cases import nearest_cases

st.set_page_config(page_title="OpenMetalAM-AI", layout="wide")
st.title("OpenMetalAM-AI")
st.caption("Literature-mined, metallurgy-aware ML for metal additive manufacturing")

TARGETS = [
    "yield_strength_MPa",
    "UTS_MPa",
    "elongation_percent",
    "hardness_HV",
    "elastic_modulus_GPa",
    "surface_roughness_Ra_um",
    "relative_density_percent",
]

DEFAULT_FEATURES = [
    "alloy",
    "AM_process",
    "AM_subprocess",
    "machine",
    "laser_power_W",
    "scan_speed_mm_s",
    "hatch_spacing_um",
    "layer_thickness_um",
    "beam_diameter_um",
    "build_orientation",
    "specimen_orientation",
    "heat_treatment",
    "relative_density_percent",
    "porosity_percent",
    "grain_size_um",
    "grain_morphology",
    "texture_intensity_MRD",
    "matrix_phase",
    "secondary_phases",
    "linear_energy_density_J_mm",
    "volumetric_energy_density_J_mm3",
    "beam_power_density_W_mm2",
]


def load_default_data() -> pd.DataFrame:
    demo = REPO_ROOT / "data" / "examples" / "demo_training_data.csv"
    template = REPO_ROOT / "data" / "examples" / "user_input_template.csv"
    if demo.exists():
        return pd.read_csv(demo)
    if template.exists():
        return pd.read_csv(template)
    return pd.DataFrame()


def safe_build_features(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return build_all_features(df.copy())


with st.sidebar:
    st.header("Data")
    uploaded_train = st.file_uploader("Upload training dataset CSV", type=["csv"], key="train")
    page = st.radio(
        "Page",
        [
            "1 Dataset explorer",
            "2 Train benchmark",
            "3 Predict user input",
            "4 Data requirements",
        ],
    )

try:
    raw_train = pd.read_csv(uploaded_train) if uploaded_train is not None else load_default_data()
except Exception as exc:
    st.error(f"Could not read training CSV: {exc}")
    raw_train = load_default_data()

train_df = safe_build_features(raw_train)

if train_df.empty:
    st.error("No dataset loaded. Upload a CSV or add data/examples/demo_training_data.csv.")
    st.stop()

if page == "1 Dataset explorer":
    st.subheader("Dataset")
    st.info("Using uploaded CSV if provided; otherwise using the bundled demo dataset.")
    c1, c2 = st.columns(2)
    c1.metric("Rows", len(train_df))
    c2.metric("Columns", len(train_df.columns))
    st.dataframe(train_df, use_container_width=True)

    st.subheader("Missing values")
    miss = train_df.isna().mean().sort_values(ascending=False).rename("missing_fraction")
    st.dataframe(miss.to_frame(), use_container_width=True)

elif page == "2 Train benchmark":
    st.subheader("Train benchmark models")
    available_targets = [t for t in TARGETS if t in train_df.columns and train_df[t].notna().sum() >= 3]
    if not available_targets:
        st.warning("Upload a dataset with at least one target column and at least three non-missing labels.")
        st.stop()

    target = st.selectbox("Target", available_targets)
    feature_cols = [c for c in DEFAULT_FEATURES if c in train_df.columns and c != target]
    st.write("Using features:", feature_cols)

    n_splits = st.slider("Cross-validation folds", min_value=2, max_value=min(5, train_df[target].notna().sum()), value=min(3, train_df[target].notna().sum()))

    if st.button("Run benchmark"):
        try:
            result = benchmark_models(train_df, target, feature_cols, n_splits=n_splits)
            st.dataframe(result, use_container_width=True)
        except Exception as exc:
            st.error(f"Benchmark failed: {exc}")

elif page == "3 Predict user input":
    st.subheader("Predict with user input")
    available_targets = [t for t in TARGETS if t in train_df.columns and train_df[t].notna().sum() >= 3]
    if not available_targets:
        st.warning("Training dataset needs at least three non-missing labels for one target.")
        st.stop()

    target = st.selectbox("Target to train", available_targets)
    uploaded_query = st.file_uploader("Upload query/input CSV", type=["csv"], key="query")

    try:
        if uploaded_query is not None:
            query_raw = pd.read_csv(uploaded_query)
        else:
            query_raw = raw_train.head(1).drop(columns=[c for c in TARGETS if c in raw_train.columns], errors="ignore")
    except Exception as exc:
        st.error(f"Could not read query CSV: {exc}")
        st.stop()

    query_df = safe_build_features(query_raw)
    st.write("Input rows")
    st.dataframe(query_df, use_container_width=True)

    if st.button("Train and predict"):
        feature_cols = [
            c for c in DEFAULT_FEATURES
            if c in train_df.columns and c in query_df.columns and c != target
        ]
        if not feature_cols:
            st.error("No usable feature columns overlap between training data and query data.")
            st.stop()

        try:
            model = train_best_model(train_df, target, feature_cols, model_name="random_forest")
            preds = model.predict(query_df[feature_cols])
            out = query_df.copy()
            out[f"predicted_{target}"] = preds
            st.subheader("Predictions")
            st.dataframe(out, use_container_width=True)

            warnings = domain_report(train_df, query_df)
            st.subheader("Warnings")
            if warnings:
                for warning in warnings:
                    st.warning(warning)
            else:
                st.success("No basic out-of-domain warnings detected.")

            st.subheader("Nearest literature/training cases")
            cases = nearest_cases(
                train_df.dropna(subset=[target]),
                query_df,
                feature_cols,
                k=min(5, len(train_df.dropna(subset=[target]))),
            )
            st.dataframe(cases, use_container_width=True)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

else:
    st.subheader("Data requirements")
    st.markdown(
        """
Minimal columns for process-property prediction:

- `alloy`
- `AM_process`
- `laser_power_W`
- `scan_speed_mm_s`
- `layer_thickness_um`
- one target such as `yield_strength_MPa` or `UTS_MPa`

Stronger columns:

- `hatch_spacing_um`
- `heat_treatment`
- `specimen_orientation`
- `relative_density_percent`
- `porosity_percent`
- `grain_size_um`
- `grain_morphology`
- `matrix_phase`
- `secondary_phases`
- `texture_intensity_MRD`

Run locally with:

```bash
pip install -r requirements.txt
streamlit run src/app/streamlit_app.py
```

On Streamlit Cloud, set the main file path to:

```text
src/app/streamlit_app.py
```
        """
    )
