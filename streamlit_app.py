from __future__ import annotations

import os
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="OpenMetalAM-AI", layout="wide", page_icon="⚙️")

ROOT = Path(__file__).resolve().parent
DEMO_PATH = ROOT / "data" / "examples" / "metal_am_demo_360.csv"

TARGETS = [
    "yield_strength_MPa", "UTS_MPa", "elongation_percent", "hardness_HV",
    "elastic_modulus_GPa", "surface_roughness_Ra_um", "relative_density_percent"
]
CORE_COLS = [
    "sample_id", "alloy", "alloy_family", "AM_subprocess", "heat_treatment", "laser_power_W",
    "scan_speed_mm_s", "hatch_spacing_um", "layer_thickness_um", "volumetric_energy_density_J_mm3",
    "relative_density_percent", "porosity_percent", "grain_size_um", "grain_morphology",
    "matrix_phase", "secondary_phases", "yield_strength_MPa", "UTS_MPa", "elongation_percent", "hardness_HV"
]

PARAMETER_DICTIONARY = pd.DataFrame([
    ["alloy", "Material/alloy name", "316L, Ti6Al4V, IN718"],
    ["AM_subprocess", "AM process subtype", "L-PBF, E-PBF, L-DED, WAAM"],
    ["laser_power_W", "Laser/electron/arc heat-source power", "W"],
    ["scan_speed_mm_s", "Travel/scan speed of heat source", "mm/s"],
    ["hatch_spacing_um", "Distance between adjacent scan tracks", "µm"],
    ["layer_thickness_um", "Powder/deposited layer thickness", "µm"],
    ["volumetric_energy_density_J_mm3", "P/(v*h*t). Useful but not sufficient alone", "J/mm³"],
    ["relative_density_percent", "Measured or reported part density", "%"],
    ["porosity_percent", "Total pore volume fraction", "%"],
    ["grain_size_um", "Mean grain size or equivalent reported size", "µm"],
    ["grain_morphology", "Main microstructure morphology", "cellular, columnar, equiaxed"],
    ["texture_intensity_MRD", "EBSD/XRD texture intensity", "MRD"],
    ["matrix_phase", "Primary phase matrix", "austenite, gamma, alpha/beta, Cu"],
    ["secondary_phases", "Important phases/precipitates", "Laves, carbides, gamma prime, Si network"],
    ["yield_strength_MPa", "0.2% yield strength", "MPa"],
    ["UTS_MPa", "Ultimate tensile strength", "MPa"],
    ["elongation_percent", "Ductility/elongation to failure", "%"],
    ["hardness_HV", "Vickers hardness", "HV"],
], columns=["Column", "Meaning", "Typical unit/example"])


def load_data(uploaded) -> pd.DataFrame:
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.success("Using uploaded CSV.")
    else:
        df = pd.read_csv(DEMO_PATH)
        st.info("Using bundled 360-row demo dataset. Replace it with real literature/user data for real research use.")
    return add_features(df)


def safe_div(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return np.where((b == 0) | pd.isna(b), np.nan, a / b)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Standardize common aliases
    aliases = {"power_W":"laser_power_W", "speed_mm_s":"scan_speed_mm_s", "layer_um":"layer_thickness_um", "hatch_um":"hatch_spacing_um", "YS_MPa":"yield_strength_MPa"}
    for old, new in aliases.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    for c in ["laser_power_W", "scan_speed_mm_s", "hatch_spacing_um", "layer_thickness_um", "beam_diameter_um"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if {"laser_power_W", "scan_speed_mm_s"}.issubset(df.columns):
        df["linear_energy_density_J_mm"] = safe_div(df["laser_power_W"], df["scan_speed_mm_s"])
    if {"laser_power_W", "scan_speed_mm_s", "hatch_spacing_um", "layer_thickness_um"}.issubset(df.columns):
        h_mm = pd.to_numeric(df["hatch_spacing_um"], errors="coerce") / 1000.0
        t_mm = pd.to_numeric(df["layer_thickness_um"], errors="coerce") / 1000.0
        denom = pd.to_numeric(df["scan_speed_mm_s"], errors="coerce") * h_mm * t_mm
        df["volumetric_energy_density_J_mm3"] = np.where((denom == 0) | pd.isna(denom), np.nan, df["laser_power_W"] / denom)
    if {"laser_power_W", "beam_diameter_um"}.issubset(df.columns):
        r_mm = pd.to_numeric(df["beam_diameter_um"], errors="coerce") / 2000.0
        area = math.pi * r_mm * r_mm
        df["beam_power_density_W_mm2"] = np.where((area == 0) | pd.isna(area), np.nan, df["laser_power_W"] / area)
    if {"thermal_conductivity_W_mK", "density_g_cm3", "specific_heat_J_kgK"}.issubset(df.columns):
        rho = pd.to_numeric(df["density_g_cm3"], errors="coerce") * 1000.0
        cp = pd.to_numeric(df["specific_heat_J_kgK"], errors="coerce")
        k = pd.to_numeric(df["thermal_conductivity_W_mK"], errors="coerce")
        df["thermal_diffusivity_m2_s"] = np.where((rho*cp == 0) | pd.isna(rho*cp), np.nan, k / (rho * cp))
    return df


def feature_columns(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    exclude = set(TARGETS + ["sample_id", "source_doi", target])
    cols = [c for c in df.columns if c not in exclude]
    num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in cols if c not in num]
    return num, cat


def build_pipeline(df: pd.DataFrame, target: str, model_name: str = "Random Forest") -> Pipeline:
    num, cat = feature_columns(df, target)
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, cat)])
    if model_name == "Gradient Boosting":
        model = GradientBoostingRegressor(random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=2)
    return Pipeline([("preprocess", pre), ("model", model)])


def evaluate(df: pd.DataFrame, target: str, model_name: str) -> dict:
    work = df.dropna(subset=[target]).copy()
    if len(work) < 8:
        return {"rows": len(work), "r2": np.nan, "mae": np.nan}
    pipe = build_pipeline(work, target, model_name)
    X = work.drop(columns=[target])
    y = work[target]
    folds = min(5, len(work))
    cv = KFold(n_splits=folds, shuffle=True, random_state=42)
    pred = cross_val_predict(pipe, X, y, cv=cv)
    return {"rows": len(work), "r2": r2_score(y, pred), "mae": mean_absolute_error(y, pred)}


def nearest_cases(train: pd.DataFrame, query: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    common = [c for c in ["laser_power_W","scan_speed_mm_s","hatch_spacing_um","layer_thickness_um","relative_density_percent","porosity_percent","grain_size_um","volumetric_energy_density_J_mm3"] if c in train.columns and c in query.columns]
    if not common or len(train) < 2:
        return pd.DataFrame()
    X = train[common].apply(pd.to_numeric, errors="coerce").fillna(train[common].median(numeric_only=True))
    q = query[common].apply(pd.to_numeric, errors="coerce").fillna(train[common].median(numeric_only=True))
    nbrs = NearestNeighbors(n_neighbors=min(n, len(train))).fit(X)
    distances, idx = nbrs.kneighbors(q.iloc[[0]])
    out = train.iloc[idx[0]].copy()
    out.insert(0, "distance_score", np.round(distances[0], 3))
    return out


def warnings_for_query(train: pd.DataFrame, q: pd.DataFrame) -> List[str]:
    warnings = []
    row = q.iloc[0]
    for c in ["alloy", "AM_subprocess", "heat_treatment"]:
        if c in train.columns and c in q.columns and pd.notna(row.get(c)):
            if str(row[c]) not in set(train[c].dropna().astype(str)):
                warnings.append(f"{c}='{row[c]}' was not seen in training data.")
    for c in ["laser_power_W", "scan_speed_mm_s", "hatch_spacing_um", "layer_thickness_um", "volumetric_energy_density_J_mm3"]:
        if c in train.columns and c in q.columns and pd.notna(row.get(c)):
            vals = pd.to_numeric(train[c], errors="coerce").dropna()
            if len(vals) > 0 and (float(row[c]) < vals.min() or float(row[c]) > vals.max()):
                warnings.append(f"{c}={row[c]} is outside training range [{vals.min():.3g}, {vals.max():.3g}].")
    return warnings


def card(title: str, value: str, note: str = ""):
    st.markdown(f"""
    <div style='border:1px solid #333;border-radius:14px;padding:18px;margin:6px 0;background:#111827'>
      <div style='font-size:0.95rem;color:#9ca3af'>{title}</div>
      <div style='font-size:1.8rem;font-weight:700;color:#f9fafb'>{value}</div>
      <div style='font-size:0.85rem;color:#9ca3af'>{note}</div>
    </div>
    """, unsafe_allow_html=True)


st.title("OpenMetalAM-AI")
st.caption("Stable Streamlit version — metal AM process → microstructure/phases → property intelligence")

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload training dataset CSV", type=["csv"])
    page = st.radio("Page", ["1 Overview", "2 Data cockpit", "3 Alloy passport", "4 Train benchmark", "5 Predict / user data", "6 Microstructure + phases", "7 Data dictionary"])

df = load_data(uploaded)

if page == "1 Overview":
    st.header("Dataset coverage snapshot")
    c1,c2,c3,c4 = st.columns(4)
    with c1: card("Rows", str(len(df)), "training/examples")
    with c2: card("Alloys", str(df.get("alloy", pd.Series(dtype=str)).nunique()), "unique materials")
    with c3: card("AM subprocesses", str(df.get("AM_subprocess", pd.Series(dtype=str)).nunique()), "LPBF, DED, etc.")
    with c4: card("Heat treatments", str(df.get("heat_treatment", pd.Series(dtype=str)).nunique()), "post-processing states")
    st.subheader("What this app is supposed to do")
    st.write("It lets users inspect AM metallurgy data, understand alloy/process parameters, train baseline models, predict properties, and include microstructure/phase descriptors when available.")
    st.warning("Bundled data is demo-style. For real science, upload curated literature or lab data with source DOI and quality scores.")
    show = [c for c in CORE_COLS if c in df.columns]
    st.subheader("Readable preview")
    st.dataframe(df[show].head(30), use_container_width=True)

elif page == "2 Data cockpit":
    st.header("Data cockpit")
    left,right = st.columns([1,3])
    with left:
        alloys = sorted(df.get("alloy", pd.Series(dtype=str)).dropna().unique())
        subs = sorted(df.get("AM_subprocess", pd.Series(dtype=str)).dropna().unique())
        hts = sorted(df.get("heat_treatment", pd.Series(dtype=str)).dropna().unique())
        q = sorted(df.get("quality_score", pd.Series(dtype=str)).dropna().unique())
        sel_alloy = st.multiselect("Alloy", alloys, default=alloys[:5])
        sel_sub = st.multiselect("AM subprocess", subs, default=subs)
        sel_ht = st.multiselect("Heat treatment", hts, default=hts)
        sel_q = st.multiselect("Quality", q, default=q)
    view = df.copy()
    if sel_alloy: view = view[view["alloy"].isin(sel_alloy)]
    if sel_sub: view = view[view["AM_subprocess"].isin(sel_sub)]
    if sel_ht: view = view[view["heat_treatment"].isin(sel_ht)]
    if sel_q and "quality_score" in view: view = view[view["quality_score"].isin(sel_q)]
    with right:
        card("Filtered rows", str(len(view)), "after current filters")
        cols = [c for c in CORE_COLS if c in view.columns]
        st.dataframe(view[cols], use_container_width=True, height=520)

elif page == "3 Alloy passport":
    st.header("Alloy passport")
    alloy = st.selectbox("Choose alloy", sorted(df["alloy"].dropna().unique()))
    sub = df[df["alloy"] == alloy]
    row = sub.iloc[0]
    c1,c2,c3,c4 = st.columns(4)
    with c1: card("Family", str(row.get("alloy_family", "—")))
    with c2: card("Base element", str(row.get("base_element", "—")))
    with c3: card("Rows", str(len(sub)))
    with c4: card("Matrix phase", str(row.get("matrix_phase", "—")))
    st.subheader("Material descriptors")
    desc_cols = [c for c in ["density_g_cm3","melting_temp_C","thermal_conductivity_W_mK","specific_heat_J_kgK","CTE_1e6_K","secondary_phases","microstructure_notes"] if c in sub.columns]
    st.table(pd.DataFrame({"descriptor": desc_cols, "value": [row.get(c) for c in desc_cols]}))
    st.subheader("Property summary for this alloy")
    props = [t for t in TARGETS if t in sub.columns]
    st.dataframe(sub.groupby("heat_treatment")[props].agg(["count","mean","min","max"]).round(2), use_container_width=True)

elif page == "4 Train benchmark":
    st.header("Train benchmark")
    target = st.selectbox("Target property", [t for t in TARGETS if t in df.columns])
    results=[]
    for model_name in ["Random Forest", "Gradient Boosting"]:
        try:
            results.append({"model":model_name, **evaluate(df,target,model_name)})
        except Exception as e:
            results.append({"model":model_name,"rows":0,"r2":np.nan,"mae":np.nan,"error":str(e)})
    st.dataframe(pd.DataFrame(results).round(4), use_container_width=True)
    st.info("Use source-paper split and leave-one-alloy-out validation later for real benchmarking. Random CV is only a first check.")

elif page == "5 Predict / user data":
    st.header("Predict / user data")
    target = st.selectbox("Target", [t for t in TARGETS if t in df.columns])
    mode = st.radio("Input mode", ["Manual single input", "Upload query CSV"])
    if mode == "Upload query CSV":
        qfile = st.file_uploader("Upload query CSV", type=["csv"], key="query")
        if qfile is None:
            st.stop()
        query = add_features(pd.read_csv(qfile))
    else:
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            alloy = st.selectbox("Alloy", sorted(df["alloy"].dropna().unique()))
            sp = st.selectbox("AM subprocess", sorted(df["AM_subprocess"].dropna().unique()))
        with c2:
            ht = st.selectbox("Heat treatment", sorted(df["heat_treatment"].dropna().unique()))
            gm = st.selectbox("Grain morphology", sorted(df["grain_morphology"].dropna().unique()))
        with c3:
            P = st.number_input("Power W", 50.0, 6000.0, 250.0)
            v = st.number_input("Scan speed mm/s", 1.0, 3000.0, 900.0)
        with c4:
            h = st.number_input("Hatch spacing µm", 10.0, 3000.0, 100.0)
            t = st.number_input("Layer thickness µm", 5.0, 2500.0, 30.0)
        query = pd.DataFrame([{
            "sample_id":"query_001","alloy":alloy,"AM_subprocess":sp,"AM_process":"PBF" if sp in ["L-PBF","E-PBF"] else "DED",
            "heat_treatment":ht,"grain_morphology":gm,"laser_power_W":P,"scan_speed_mm_s":v,"hatch_spacing_um":h,"layer_thickness_um":t,
            "relative_density_percent":st.number_input("Relative density %", 80.0, 100.0, 99.0),
            "porosity_percent":st.number_input("Porosity %", 0.0, 20.0, 0.5),
            "grain_size_um":st.number_input("Grain size µm", 0.1, 500.0, 20.0),
            "texture_intensity_MRD":st.number_input("Texture intensity MRD", 0.0, 20.0, 3.0),
            "matrix_phase": df[df["alloy"]==alloy]["matrix_phase"].iloc[0] if "matrix_phase" in df.columns else "unknown",
            "secondary_phases": df[df["alloy"]==alloy]["secondary_phases"].iloc[0] if "secondary_phases" in df.columns else "unknown"
        }])
        query = add_features(query)
    train = df.dropna(subset=[target]).copy()
    pipe = build_pipeline(train, target, "Random Forest")
    X = train.drop(columns=[target])
    y = train[target]
    pipe.fit(X, y)
    preds = pipe.predict(query)
    st.subheader("Prediction")
    card(f"Predicted {target}", f"{preds[0]:.2f}", "research guidance, not certification")
    warns = warnings_for_query(train, query)
    if warns:
        st.warning("\n".join([f"- {w}" for w in warns]))
    else:
        st.success("No basic out-of-domain warning triggered.")
    st.subheader("Query features")
    st.dataframe(query[[c for c in query.columns if c not in TARGETS]], use_container_width=True)
    st.subheader("Nearest training/literature-style cases")
    near = nearest_cases(train, query, 8)
    if len(near):
        st.dataframe(near[[c for c in ["distance_score"]+CORE_COLS if c in near.columns]], use_container_width=True)

elif page == "6 Microstructure + phases":
    st.header("Microstructure + phases")
    st.write("This page exposes the V2/V3 idea: do not predict properties only from process parameters; include grain morphology, porosity, phases and texture when available.")
    cols = [c for c in ["alloy","AM_subprocess","heat_treatment","grain_morphology","grain_size_um","porosity_percent","relative_density_percent","texture_intensity_MRD","matrix_phase","secondary_phases","yield_strength_MPa","UTS_MPa","elongation_percent"] if c in df.columns]
    st.dataframe(df[cols].head(200), use_container_width=True, height=520)
    st.subheader("Average properties by microstructure label")
    props = [c for c in ["yield_strength_MPa","UTS_MPa","elongation_percent","hardness_HV"] if c in df.columns]
    st.dataframe(df.groupby("grain_morphology")[props].agg(["count","mean","min","max"]).round(2), use_container_width=True)

elif page == "7 Data dictionary":
    st.header("Data dictionary")
    st.dataframe(PARAMETER_DICTIONARY, use_container_width=True, height=520)
    st.subheader("Minimum useful columns for user CSV")
    st.code("""alloy, AM_subprocess, laser_power_W, scan_speed_mm_s, hatch_spacing_um, layer_thickness_um,
heat_treatment, relative_density_percent, porosity_percent,
yield_strength_MPa or UTS_MPa or elongation_percent or hardness_HV""")
