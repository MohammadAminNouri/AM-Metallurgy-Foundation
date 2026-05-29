from __future__ import annotations

from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DEMO_PATH = ROOT / "data" / "examples" / "metal_am_literature_style_demo.csv"
ALLOY_REF_PATH = ROOT / "data" / "reference" / "alloy_reference.csv"
PARAM_DICT_PATH = ROOT / "data" / "reference" / "parameter_dictionary.csv"
OPEN_SOURCES_PATH = ROOT / "data" / "reference" / "open_data_sources.csv"
TRAIN_TEMPLATE = ROOT / "data" / "examples" / "user_training_template.csv"
PRED_TEMPLATE = ROOT / "data" / "examples" / "user_prediction_template.csv"

TARGETS = [
    "yield_strength_MPa", "UTS_MPa", "elongation_percent", "hardness_HV",
    "elastic_modulus_GPa", "surface_roughness_Ra_um", "relative_density_percent"
]
CORE_NUMERIC = [
    "laser_power_W", "scan_speed_mm_s", "hatch_spacing_um", "layer_thickness_um", "beam_diameter_um",
    "density_kg_m3", "melting_C", "thermal_conductivity_W_mK", "specific_heat_J_kgK", "cte_1e6_K",
    "relative_density_percent", "porosity_percent", "grain_size_um", "texture_intensity_MRD",
    "linear_energy_density_J_mm", "volumetric_energy_density_J_mm3", "beam_power_density_W_mm2",
    "thermal_diffusivity_m2_s", "thermal_headroom_C", "porosity_risk_index", "density_quality_index"
]
CORE_CATEGORICAL = [
    "alloy", "alloy_family", "base_element", "AM_process", "AM_subprocess", "machine",
    "build_orientation", "specimen_orientation", "heat_treatment", "grain_morphology",
    "matrix_phase", "secondary_phases", "data_quality_score"
]

st.set_page_config(page_title="OpenMetalAM-AI", page_icon="⚙️", layout="wide")

CSS = """
<style>
.block-container {padding-top: 2rem; padding-bottom: 4rem; max-width: 1400px;}
.big-title {font-size: 3.1rem; font-weight: 850; line-height: 1.0; margin-bottom: 0.2rem;}
.subtitle {font-size: 1.05rem; color: #A7B0BE; margin-bottom: 1.3rem;}
.card {padding: 1.0rem 1.2rem; border-radius: 1rem; background: rgba(120,130,155,0.12); border: 1px solid rgba(150,160,180,0.18);}
.warn {padding: 0.8rem 1rem; border-radius: 0.8rem; background: rgba(255,190,80,0.14); border: 1px solid rgba(255,190,80,0.35);}
.good {padding: 0.8rem 1rem; border-radius: 0.8rem; background: rgba(80,220,150,0.12); border: 1px solid rgba(80,220,150,0.35);}
.small {font-size: 0.88rem; color: #A7B0BE;}
code {white-space: pre-wrap;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def read_csv_path(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def read_uploaded(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)

@st.cache_data(show_spinner=False)
def load_defaults():
    demo = read_csv_path(DEMO_PATH)
    alloy_ref = read_csv_path(ALLOY_REF_PATH)
    param_dict = read_csv_path(PARAM_DICT_PATH)
    open_sources = read_csv_path(OPEN_SOURCES_PATH)
    return demo, alloy_ref, param_dict, open_sources


def safe_div(num, den):
    return np.where((den == 0) | pd.isna(den), np.nan, num / den)


def enrich_features(df: pd.DataFrame, alloy_ref: pd.DataFrame | None = None) -> pd.DataFrame:
    df = df.copy()
    # Auto-join reference alloy properties if user uploaded only alloy names.
    if alloy_ref is not None and "alloy" in df.columns:
        ref_cols = [c for c in alloy_ref.columns if c not in df.columns or c in ["alloy"]]
        if len(ref_cols) > 1:
            df = df.merge(alloy_ref[ref_cols], on="alloy", how="left")
    for col in ["laser_power_W", "scan_speed_mm_s", "hatch_spacing_um", "layer_thickness_um", "beam_diameter_um"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    hatch_mm = df["hatch_spacing_um"] / 1000.0
    layer_mm = df["layer_thickness_um"] / 1000.0
    beam_mm = df["beam_diameter_um"] / 1000.0
    df["linear_energy_density_J_mm"] = safe_div(df["laser_power_W"], df["scan_speed_mm_s"])
    df["volumetric_energy_density_J_mm3"] = safe_div(df["laser_power_W"], df["scan_speed_mm_s"] * hatch_mm * layer_mm)
    area = np.pi * (beam_mm / 2) ** 2
    df["beam_power_density_W_mm2"] = safe_div(df["laser_power_W"], area)
    for col in ["thermal_conductivity_W_mK", "density_kg_m3", "specific_heat_J_kgK", "melting_C", "relative_density_percent", "porosity_percent", "grain_size_um", "texture_intensity_MRD"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["thermal_diffusivity_m2_s"] = safe_div(df["thermal_conductivity_W_mK"], df["density_kg_m3"] * df["specific_heat_J_kgK"])
    df["thermal_headroom_C"] = df["melting_C"] - 25.0
    df["porosity_risk_index"] = df["porosity_percent"].fillna(0) + np.maximum(0, 98.5 - df["relative_density_percent"].fillna(98.5))
    df["density_quality_index"] = df["relative_density_percent"].fillna(df["relative_density_percent"].median() if df["relative_density_percent"].notna().any() else 98.0) / 100.0
    for col in CORE_CATEGORICAL:
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = df[col].fillna("unknown").astype(str)
    return df


def feature_columns(df: pd.DataFrame, target: str | None = None) -> tuple[list[str], list[str]]:
    excluded = set(TARGETS + ["sample_id", "source_doi", "microstructure_notes"])
    if target:
        excluded.add(target)
    num = [c for c in CORE_NUMERIC if c in df.columns and c not in excluded]
    cat = [c for c in CORE_CATEGORICAL if c in df.columns and c not in excluded]
    return num, cat


def make_preprocessor(num_cols, cat_cols):
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", ohe)]), cat_cols),
        ],
        remainder="drop",
    )


def make_model(kind: str, num_cols, cat_cols, seed=42):
    if kind == "Random Forest":
        reg = RandomForestRegressor(n_estimators=350, min_samples_leaf=2, random_state=seed, n_jobs=-1)
    elif kind == "Gradient Boosting":
        reg = GradientBoostingRegressor(random_state=seed)
    else:
        reg = RandomForestRegressor(n_estimators=350, min_samples_leaf=2, random_state=seed, n_jobs=-1)
    return Pipeline([("preprocess", make_preprocessor(num_cols, cat_cols)), ("model", reg)])


def train_model(df: pd.DataFrame, target: str, model_kind="Random Forest"):
    data = df.dropna(subset=[target]).copy()
    if len(data) < 8:
        raise ValueError(f"Need at least 8 rows with target '{target}' for a meaningful demo training. Current rows: {len(data)}")
    num_cols, cat_cols = feature_columns(data, target)
    model = make_model(model_kind, num_cols, cat_cols)
    X, y = data[num_cols + cat_cols], data[target].astype(float)
    model.fit(X, y)
    return model, data, num_cols, cat_cols


def benchmark_models(df: pd.DataFrame, target: str):
    data = df.dropna(subset=[target]).copy()
    num_cols, cat_cols = feature_columns(data, target)
    X, y = data[num_cols + cat_cols], data[target].astype(float)
    rows = []
    folds = min(5, max(2, len(data)//10))
    cv = KFold(n_splits=folds, shuffle=True, random_state=42)
    scoring = {"mae":"neg_mean_absolute_error", "r2":"r2", "rmse":"neg_root_mean_squared_error"}
    for kind in ["Random Forest", "Gradient Boosting"]:
        model = make_model(kind, num_cols, cat_cols)
        out = cross_validate(model, X, y, cv=cv, scoring=scoring, error_score="raise")
        rows.append({
            "model": kind,
            "rows": len(data),
            "features": len(num_cols)+len(cat_cols),
            "CV folds": folds,
            "MAE": -out["test_mae"].mean(),
            "RMSE": -out["test_rmse"].mean(),
            "R2": out["test_r2"].mean(),
        })
    return pd.DataFrame(rows).sort_values("MAE")


def split_eval(df: pd.DataFrame, target: str, model_kind="Random Forest"):
    data = df.dropna(subset=[target]).copy()
    num_cols, cat_cols = feature_columns(data, target)
    X, y = data[num_cols + cat_cols], data[target].astype(float)
    if len(data) < 12:
        return None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=7)
    model = make_model(model_kind, num_cols, cat_cols)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return pd.DataFrame({"actual": yte.values, "predicted": pred, "error": pred-yte.values})


def nearest_cases(train_df: pd.DataFrame, query_df: pd.DataFrame, target: str, k=5):
    num_cols, cat_cols = feature_columns(train_df, target)
    cols = num_cols + cat_cols
    train = train_df[cols].copy()
    query = query_df[cols].copy()
    prep = make_preprocessor(num_cols, cat_cols)
    X = prep.fit_transform(train)
    Q = prep.transform(query)
    nn = NearestNeighbors(n_neighbors=min(k, len(train)), metric="euclidean")
    nn.fit(X)
    dist, idx = nn.kneighbors(Q)
    results = []
    for qi in range(len(query_df)):
        case = train_df.iloc[idx[qi]].copy()
        case["distance_score"] = dist[qi]
        results.append(case)
    return results[0] if results else pd.DataFrame()


def domain_warnings(train_df: pd.DataFrame, query_df: pd.DataFrame):
    warnings = []
    for col in ["alloy", "AM_subprocess", "heat_treatment", "grain_morphology", "matrix_phase"]:
        if col in query_df.columns and col in train_df.columns:
            unseen = sorted(set(query_df[col].dropna().astype(str)) - set(train_df[col].dropna().astype(str)))
            if unseen:
                warnings.append(f"Unseen {col}: {', '.join(unseen)}")
    for col in ["laser_power_W", "scan_speed_mm_s", "hatch_spacing_um", "layer_thickness_um", "volumetric_energy_density_J_mm3", "grain_size_um", "texture_intensity_MRD"]:
        if col in query_df.columns and col in train_df.columns and train_df[col].notna().any():
            mn, mx = train_df[col].min(), train_df[col].max()
            qmin, qmax = query_df[col].min(), query_df[col].max()
            if pd.notna(qmin) and (qmin < mn or qmax > mx):
                warnings.append(f"{col} outside training range: training {mn:.3g}–{mx:.3g}; query {qmin:.3g}–{qmax:.3g}")
    missing_key = [c for c in ["laser_power_W", "scan_speed_mm_s", "hatch_spacing_um", "layer_thickness_um"] if c in query_df.columns and query_df[c].isna().any()]
    if missing_key:
        warnings.append("Missing key process values: " + ", ".join(missing_key))
    return warnings


def download_button_for_file(path: Path, label: str):
    if path.exists():
        st.download_button(label, data=path.read_bytes(), file_name=path.name, mime="text/csv")

# Load data
base_demo, alloy_ref, param_dict, open_sources = load_defaults()

with st.sidebar:
    st.markdown("## Data source")
    uploaded = st.file_uploader("Upload your training CSV", type=["csv"])
    if uploaded is not None:
        try:
            raw_df = read_uploaded(uploaded)
            source_label = "uploaded CSV"
        except Exception as e:
            st.error(f"Could not read uploaded CSV: {e}")
            raw_df = base_demo.copy(); source_label = "bundled demo"
    else:
        raw_df = base_demo.copy(); source_label = "bundled demo"
    use_ref = st.checkbox("Auto-fill missing alloy thermal properties", value=True)
    target = st.selectbox("Target property", [t for t in TARGETS if t in raw_df.columns or t in base_demo.columns], index=0)
    page = st.radio("Page", ["1 Overview", "2 Data cockpit", "3 Alloy passport", "4 Train benchmark", "5 Predict / user data", "6 Microstructure + phases", "7 Data dictionary", "8 Sources + roadmap"])

# feature enrichment
df = enrich_features(raw_df, alloy_ref if use_ref else None)

st.markdown('<div class="big-title">OpenMetalAM-AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Metallurgy-aware AM data + prediction cockpit: process → microstructure → phase → property</div>', unsafe_allow_html=True)

# Top metric cards
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Alloys", df.get("alloy", pd.Series()).nunique())
c3.metric("Targets", sum(t in df.columns and df[t].notna().any() for t in TARGETS))
c4.metric("Numeric features", len(feature_columns(df, target)[0]))
c5.metric("Data source", source_label)
st.caption("Bundled data is a literature-style starter/demo dataset for testing the app and pipeline. Replace it with real curated literature or lab data for scientific use.")

if page == "1 Overview":
    st.subheader("What this app is supposed to do")
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("""
        This app is not just a raw table viewer. It is a **metallurgy-aware prediction cockpit**.

        **V1:** mechanical-property prediction from AM process + alloy data.  
        **V2:** microstructure, crystallography, and phase descriptors.  
        **V3:** process + microstructure + phase-aware property prediction.
        """)
        st.markdown("""
        **Current pipeline inside the app:**
        1. Load demo or user CSV.
        2. Auto-fill missing alloy reference properties where possible.
        3. Create physics-aware features: line energy, volumetric energy density, beam power density, thermal diffusivity, porosity-risk proxies.
        4. Train tree-based models.
        5. Predict selected property.
        6. Show out-of-domain warnings and nearest similar cases.
        """)
    with right:
        st.markdown("### PSP chain")
        st.code("""process parameters
  ↓
thermal/material descriptors
  ↓
melt pool + porosity + density
  ↓
microstructure + crystallography
  ↓
phases / precipitates
  ↓
YS, UTS, elongation, hardness, fatigue""")
    st.divider()
    st.subheader("Dataset coverage snapshot")
    a,b,c = st.columns(3)
    with a:
        st.write("Alloy count")
        st.bar_chart(df["alloy"].value_counts())
    with b:
        st.write("AM subprocess count")
        st.bar_chart(df["AM_subprocess"].value_counts())
    with c:
        st.write("Heat treatment count")
        st.bar_chart(df["heat_treatment"].value_counts())

elif page == "2 Data cockpit":
    st.subheader("Data cockpit — readable, filtered, explained")
    cols = st.columns(4)
    alloy_filter = cols[0].multiselect("Alloy", sorted(df["alloy"].unique()), default=[])
    proc_filter = cols[1].multiselect("AM subprocess", sorted(df["AM_subprocess"].unique()), default=[])
    ht_filter = cols[2].multiselect("Heat treatment", sorted(df["heat_treatment"].unique()), default=[])
    q_filter = cols[3].multiselect("Data quality", sorted(df["data_quality_score"].unique()) if "data_quality_score" in df else [], default=[])
    view = df.copy()
    if alloy_filter: view = view[view["alloy"].isin(alloy_filter)]
    if proc_filter: view = view[view["AM_subprocess"].isin(proc_filter)]
    if ht_filter: view = view[view["heat_treatment"].isin(ht_filter)]
    if q_filter: view = view[view["data_quality_score"].isin(q_filter)]
    st.markdown("### Main process + structure + property table")
    shown_cols = [c for c in ["sample_id","alloy","AM_subprocess","heat_treatment","laser_power_W","scan_speed_mm_s","hatch_spacing_um","layer_thickness_um","volumetric_energy_density_J_mm3","relative_density_percent","porosity_percent","grain_morphology","matrix_phase",target] if c in view.columns]
    st.dataframe(view[shown_cols], use_container_width=True, height=430)
    st.markdown("### Missing values by column")
    miss = view.isna().sum().sort_values(ascending=False).reset_index()
    miss.columns = ["column", "missing_count"]
    st.dataframe(miss[miss["missing_count"]>0], use_container_width=True, height=220)
    st.markdown("### Target distribution")
    if target in view.columns:
        st.bar_chart(view[target].dropna())

elif page == "3 Alloy passport":
    st.subheader("Alloy passport — see the material before trusting prediction")
    alloy = st.selectbox("Choose alloy", sorted(df["alloy"].dropna().unique()))
    ref = alloy_ref[alloy_ref["alloy"] == alloy]
    rows = df[df["alloy"] == alloy]
    if not ref.empty:
        r = ref.iloc[0]
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Family", str(r.get("family","")))
        c2.metric("Base", str(r.get("base","")))
        c3.metric("Density", f"{r.get('density_kg_m3', np.nan):.0f} kg/m³")
        c4.metric("Tm", f"{r.get('melting_C', np.nan):.0f} °C")
        c5.metric("k", f"{r.get('thermal_conductivity_W_mK', np.nan):.1f} W/mK")
        st.markdown("### Phase/microstructure expectation")
        st.info(f"Matrix phase: **{r.get('matrix_phase','')}** | Secondary phases: **{r.get('secondary_phases','')}** | Notes: {r.get('micro','')}")
    st.markdown("### Available rows for this alloy")
    cols = [c for c in ["sample_id","AM_subprocess","heat_treatment","relative_density_percent","grain_size_um","grain_morphology","matrix_phase","secondary_phases","yield_strength_MPa","UTS_MPa","elongation_percent","hardness_HV"] if c in rows.columns]
    st.dataframe(rows[cols], use_container_width=True, height=420)
    if target in rows.columns and rows[target].notna().any():
        st.markdown(f"### {target} by heat treatment")
        st.bar_chart(rows.groupby("heat_treatment")[target].mean().sort_values())

elif page == "4 Train benchmark":
    st.subheader("Train benchmark — check whether the dataset supports prediction")
    st.markdown("This trains models on the current dataset and selected target. For real work, use source-paper split and alloy-family split later, not only random CV.")
    if target not in df.columns or df[target].notna().sum() < 8:
        st.error(f"Not enough rows for target: {target}")
    else:
        with st.spinner("Training benchmark models..."):
            bench = benchmark_models(df, target)
            eval_df = split_eval(df, target)
        st.dataframe(bench.style.format({"MAE":"{:.3f}","RMSE":"{:.3f}","R2":"{:.3f}"}), use_container_width=True)
        if eval_df is not None:
            st.markdown("### Actual vs predicted holdout preview")
            st.scatter_chart(eval_df, x="actual", y="predicted")
            st.dataframe(eval_df.head(30), use_container_width=True)
        st.markdown("### Feature groups used")
        num_cols, cat_cols = feature_columns(df, target)
        st.write("Numeric:", num_cols)
        st.write("Categorical:", cat_cols)

elif page == "5 Predict / user data":
    st.subheader("Predict / user data — upload query or fill one sample")
    st.markdown("Train from the current dataset, then predict for one or more new rows. You can upload a query CSV or use the form below.")
    model_kind = st.selectbox("Model", ["Random Forest", "Gradient Boosting"])
    query_upload = st.file_uploader("Optional: upload prediction/query CSV without target column", type=["csv"], key="query")
    if query_upload:
        query_raw = read_uploaded(query_upload)
    else:
        st.markdown("### Manual sample form")
        c1,c2,c3 = st.columns(3)
        alloy = c1.selectbox("Alloy", sorted(alloy_ref["alloy"].unique()))
        proc = c2.selectbox("AM subprocess", ["L-PBF","E-PBF","L-DED","WAAM"])
        ht = c3.selectbox("Heat treatment", ["as-built","stress relieved","annealed","HIP","aged","solution+aged"])
        c4,c5,c6,c7 = st.columns(4)
        power = c4.number_input("laser_power_W", value=250.0, min_value=0.0)
        speed = c5.number_input("scan_speed_mm_s", value=900.0, min_value=0.0)
        hatch = c6.number_input("hatch_spacing_um", value=100.0, min_value=0.0)
        layer = c7.number_input("layer_thickness_um", value=30.0, min_value=0.0)
        c8,c9,c10,c11 = st.columns(4)
        density = c8.number_input("relative_density_percent", value=99.0, min_value=0.0, max_value=100.0)
        porosity = c9.number_input("porosity_percent", value=1.0, min_value=0.0, max_value=100.0)
        grain = c10.number_input("grain_size_um", value=20.0, min_value=0.0)
        texture = c11.number_input("texture_intensity_MRD", value=2.0, min_value=0.0)
        gm = st.selectbox("grain_morphology", ["unknown","cellular","columnar","dendritic","fine equiaxed","mixed columnar-equiaxed","martensitic"])
        refrow = alloy_ref[alloy_ref["alloy"] == alloy].iloc[0].to_dict()
        query_raw = pd.DataFrame([{
            "sample_id":"user_query_001", "alloy":alloy, "AM_process":"PBF" if proc in ["L-PBF","E-PBF"] else "DED", "AM_subprocess":proc,
            "machine":"user/system", "laser_power_W":power, "scan_speed_mm_s":speed, "hatch_spacing_um":hatch, "layer_thickness_um":layer, "beam_diameter_um":80,
            "specimen_orientation":"horizontal", "heat_treatment":ht, "relative_density_percent":density, "porosity_percent":porosity, "grain_size_um":grain,
            "grain_morphology":gm, "texture_intensity_MRD":texture, "matrix_phase":refrow.get("matrix_phase","unknown"), "secondary_phases":refrow.get("secondary_phases","unknown")
        }])
    query = enrich_features(query_raw, alloy_ref if use_ref else None)
    st.markdown("### Query preview")
    st.dataframe(query, use_container_width=True, height=180)
    if st.button("Train and predict", type="primary"):
        try:
            model, train_data, num_cols, cat_cols = train_model(df, target, model_kind)
            query_X = query[num_cols + cat_cols]
            pred = model.predict(query_X)
            out = query[[c for c in ["sample_id","alloy","AM_subprocess","heat_treatment","grain_morphology","matrix_phase"] if c in query.columns]].copy()
            out[f"predicted_{target}"] = pred
            st.success("Prediction complete")
            st.dataframe(out, use_container_width=True)
            warns = domain_warnings(train_data, query)
            if warns:
                st.markdown("### Trust warnings")
                for w in warns:
                    st.markdown(f"<div class='warn'>⚠️ {w}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='good'>No major out-of-domain warning from the current checks.</div>", unsafe_allow_html=True)
            st.markdown("### Nearest similar training/literature-style cases")
            near = nearest_cases(train_data, query, target, k=7)
            show_cols = [c for c in ["sample_id","source_doi","alloy","AM_subprocess","heat_treatment","laser_power_W","scan_speed_mm_s","volumetric_energy_density_J_mm3","relative_density_percent","grain_morphology","matrix_phase",target,"distance_score"] if c in near.columns]
            st.dataframe(near[show_cols], use_container_width=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    st.markdown("### Download templates")
    d1,d2 = st.columns(2)
    with d1: download_button_for_file(TRAIN_TEMPLATE, "Download training template")
    with d2: download_button_for_file(PRED_TEMPLATE, "Download prediction template")

elif page == "6 Microstructure + phases":
    st.subheader("Microstructure + crystallography + phases")
    st.markdown("This page shows the V2/V3 part: how structure and phase descriptors enter property prediction.")
    structure_cols = [c for c in ["alloy","AM_subprocess","heat_treatment","relative_density_percent","porosity_percent","grain_size_um","grain_morphology","texture_intensity_MRD","matrix_phase","secondary_phases",target] if c in df.columns]
    st.dataframe(df[structure_cols].head(200), use_container_width=True, height=330)
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("### Grain morphology")
        st.bar_chart(df["grain_morphology"].value_counts())
    with c2:
        st.markdown("### Matrix phase")
        st.bar_chart(df["matrix_phase"].value_counts())
    with c3:
        st.markdown("### Data quality")
        st.bar_chart(df["data_quality_score"].value_counts())
    st.markdown("### Microstructure-aware comparison idea")
    st.code("""Model A: process-only
  alloy + power + speed + hatch + layer + heat treatment → property

Model B: V3 microstructure-aware
  process + alloy + density + porosity + grain size + texture + phases → property

Compare MAE/R². If Model B improves, structure/phase descriptors are valuable.""")

elif page == "7 Data dictionary":
    st.subheader("Data dictionary — what the columns mean")
    st.dataframe(param_dict, use_container_width=True, height=460)
    st.markdown("### Required minimum columns")
    st.code("""Minimum for process-only prediction:
alloy, AM_subprocess, laser_power_W, scan_speed_mm_s, hatch_spacing_um, layer_thickness_um, heat_treatment

Better for V3 prediction:
+ relative_density_percent, porosity_percent, grain_size_um, grain_morphology, texture_intensity_MRD, matrix_phase, secondary_phases

Targets for training:
yield_strength_MPa, UTS_MPa, elongation_percent, hardness_HV, elastic_modulus_GPa""")

elif page == "8 Sources + roadmap":
    st.subheader("Open-source integrations and roadmap")
    st.dataframe(open_sources, use_container_width=True)
    st.markdown("### What is still needed to become top-level")
    st.markdown("""
    1. Replace demo data with curated literature rows.
    2. Add importers for MeltpoolNet and NIST AM-Bench where licensing allows.
    3. Add SHAP plots for model explanations.
    4. Add source-paper split and leave-one-alloy-out validation.
    5. Add microstructure-aware model comparison.
    6. Add real SEM/EBSD/XRD image modules later.
    """)

