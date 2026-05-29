"""
OpenMetalAM-AI stable Streamlit app with CSV upload + manual training data entry.
Root entrypoint for Streamlit Cloud.
"""
from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="OpenMetalAM-AI", layout="wide")

TARGETS = [
    "yield_strength_MPa",
    "UTS_MPa",
    "elongation_percent",
    "hardness_HV",
    "elastic_modulus_GPa",
    "surface_roughness_Ra_um",
    "relative_density_percent",
]

CATEGORICAL_COLS = [
    "alloy",
    "alloy_family",
    "base_element",
    "AM_process",
    "AM_subprocess",
    "machine",
    "build_orientation",
    "specimen_orientation",
    "heat_treatment",
    "grain_morphology",
    "matrix_phase",
    "secondary_phases",
    "data_quality",
]

NUMERIC_COLS_BASE = [
    "laser_power_W",
    "scan_speed_mm_s",
    "hatch_spacing_um",
    "layer_thickness_um",
    "beam_diameter_um",
    "density_g_cm3",
    "melting_temp_C",
    "thermal_conductivity_W_mK",
    "specific_heat_J_kgK",
    "CTE_1e6_K",
    "relative_density_percent",
    "porosity_percent",
    "grain_size_um",
    "texture_intensity_MRD",
]

ALL_COLUMNS = [
    "sample_id",
    "source_doi",
    "alloy",
    "alloy_family",
    "base_element",
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
    "density_g_cm3",
    "melting_temp_C",
    "thermal_conductivity_W_mK",
    "specific_heat_J_kgK",
    "CTE_1e6_K",
    "relative_density_percent",
    "porosity_percent",
    "grain_size_um",
    "grain_morphology",
    "texture_intensity_MRD",
    "matrix_phase",
    "secondary_phases",
    "data_quality",
] + TARGETS

THERMAL_DB: Dict[str, Dict[str, object]] = {
    "316L": dict(alloy_family="stainless steel", base_element="Fe", density_g_cm3=7.99, melting_temp_C=1385, thermal_conductivity_W_mK=16.2, specific_heat_J_kgK=500, CTE_1e6_K=16.0, matrix_phase="austenite", secondary_phases="delta ferrite/carbides possible"),
    "17-4PH": dict(alloy_family="precipitation hardened steel", base_element="Fe", density_g_cm3=7.80, melting_temp_C=1420, thermal_conductivity_W_mK=16.0, specific_heat_J_kgK=460, CTE_1e6_K=10.4, matrix_phase="martensite", secondary_phases="Cu precipitates"),
    "Ti6Al4V": dict(alloy_family="titanium alloy", base_element="Ti", density_g_cm3=4.43, melting_temp_C=1660, thermal_conductivity_W_mK=7.1, specific_heat_J_kgK=560, CTE_1e6_K=8.6, matrix_phase="alpha/beta", secondary_phases="alpha-prime possible"),
    "Ti6Al4V ELI": dict(alloy_family="titanium alloy", base_element="Ti", density_g_cm3=4.42, melting_temp_C=1650, thermal_conductivity_W_mK=7.0, specific_heat_J_kgK=526, CTE_1e6_K=8.5, matrix_phase="alpha/beta", secondary_phases="alpha-prime possible"),
    "IN718": dict(alloy_family="nickel superalloy", base_element="Ni", density_g_cm3=8.19, melting_temp_C=1298, thermal_conductivity_W_mK=9.0, specific_heat_J_kgK=435, CTE_1e6_K=14.0, matrix_phase="gamma", secondary_phases="gamma prime/gamma double prime/Laves"),
    "IN625": dict(alloy_family="nickel superalloy", base_element="Ni", density_g_cm3=8.44, melting_temp_C=1320, thermal_conductivity_W_mK=9.2, specific_heat_J_kgK=429, CTE_1e6_K=12.8, matrix_phase="gamma", secondary_phases="carbides/Laves possible"),
    "AlSi10Mg": dict(alloy_family="aluminum alloy", base_element="Al", density_g_cm3=2.68, melting_temp_C=580, thermal_conductivity_W_mK=160, specific_heat_J_kgK=910, CTE_1e6_K=20.5, matrix_phase="alpha-Al", secondary_phases="Si network/Mg2Si"),
    "AlSi7Mg": dict(alloy_family="aluminum alloy", base_element="Al", density_g_cm3=2.67, melting_temp_C=570, thermal_conductivity_W_mK=140, specific_heat_J_kgK=890, CTE_1e6_K=21.5, matrix_phase="alpha-Al", secondary_phases="Si/Mg2Si"),
    "CoCrMo": dict(alloy_family="cobalt alloy", base_element="Co", density_g_cm3=8.4, melting_temp_C=1390, thermal_conductivity_W_mK=13, specific_heat_J_kgK=450, CTE_1e6_K=14.0, matrix_phase="Co-rich", secondary_phases="carbides"),
    "Maraging Steel M300": dict(alloy_family="maraging steel", base_element="Fe", density_g_cm3=8.1, melting_temp_C=1413, thermal_conductivity_W_mK=14.2, specific_heat_J_kgK=452, CTE_1e6_K=10.3, matrix_phase="martensite", secondary_phases="Ni/Ti/Mo precipitates"),
    "CuCrZr": dict(alloy_family="copper alloy", base_element="Cu", density_g_cm3=8.9, melting_temp_C=1085, thermal_conductivity_W_mK=320, specific_heat_J_kgK=380, CTE_1e6_K=17.0, matrix_phase="Cu", secondary_phases="Cr/Zr precipitates"),
    "H13 Tool Steel": dict(alloy_family="tool steel", base_element="Fe", density_g_cm3=7.8, melting_temp_C=1427, thermal_conductivity_W_mK=28.6, specific_heat_J_kgK=460, CTE_1e6_K=10.4, matrix_phase="martensite/bainite", secondary_phases="carbides"),
}

PARAM_HELP = {
    "sample_id": "Unique sample/row ID. Example: my_sample_001.",
    "source_doi": "Paper DOI, datasheet ID, lab batch ID, or 'manual'.",
    "alloy": "Material/alloy name. Example: 316L, Ti6Al4V, IN718.",
    "AM_subprocess": "AM subprocess. Example: L-PBF, E-PBF, L-DED, WAAM.",
    "laser_power_W": "Laser/beam power in watts.",
    "scan_speed_mm_s": "Scan speed in mm/s.",
    "hatch_spacing_um": "Distance between scan tracks in micrometers.",
    "layer_thickness_um": "Powder/deposited layer thickness in micrometers.",
    "beam_diameter_um": "Laser/beam spot diameter in micrometers.",
    "heat_treatment": "As-built, stress relieved, annealed, HIP, aged, solution aged, etc.",
    "relative_density_percent": "Measured relative density, usually 95-100%.",
    "porosity_percent": "Measured porosity percentage.",
    "grain_size_um": "Average grain size if known.",
    "texture_intensity_MRD": "Texture intensity from EBSD/XRD if known, in MRD.",
    "grain_morphology": "Equiaxed, columnar, cellular, dendritic, martensitic, mixed, etc.",
    "matrix_phase": "Main phase: austenite, ferrite, alpha/beta, gamma, alpha-Al, Cu, etc.",
    "secondary_phases": "Precipitates/phases: carbides, Laves, gamma prime, Mg2Si, alpha-prime, etc.",
}


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in NUMERIC_COLS_BASE + TARGETS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    P = out.get("laser_power_W")
    v = out.get("scan_speed_mm_s")
    h_um = out.get("hatch_spacing_um")
    t_um = out.get("layer_thickness_um")
    bd_um = out.get("beam_diameter_um")
    if P is not None and v is not None:
        out["linear_energy_density_J_mm"] = P / v.replace(0, np.nan)
    if P is not None and v is not None and h_um is not None and t_um is not None:
        h_mm = h_um / 1000.0
        t_mm = t_um / 1000.0
        out["volumetric_energy_density_J_mm3"] = P / (v.replace(0, np.nan) * h_mm.replace(0, np.nan) * t_mm.replace(0, np.nan))
    if P is not None and bd_um is not None:
        radius_mm = (bd_um / 1000.0) / 2.0
        out["beam_power_density_W_mm2"] = P / (math.pi * radius_mm.replace(0, np.nan) ** 2)
    if {"thermal_conductivity_W_mK", "density_g_cm3", "specific_heat_J_kgK"}.issubset(out.columns):
        rho = out["density_g_cm3"] * 1000.0
        cp = out["specific_heat_J_kgK"]
        k = out["thermal_conductivity_W_mK"]
        out["thermal_diffusivity_m2_s"] = k / (rho.replace(0, np.nan) * cp.replace(0, np.nan))
    if "melting_temp_C" in out.columns:
        out["thermal_headroom_C"] = out["melting_temp_C"] - 25.0
    return out


def auto_fill_thermal(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "alloy" not in out.columns:
        return out
    for i, alloy in out["alloy"].astype(str).items():
        props = THERMAL_DB.get(alloy)
        if not props:
            continue
        for key, value in props.items():
            if key not in out.columns:
                out[key] = np.nan
            current = out.at[i, key]
            if pd.isna(current) or str(current).strip() == "":
                out.at[i, key] = value
    return out


def make_demo_dataset(n_per_alloy: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    alloys = list(THERMAL_DB.keys())
    rows = []
    ht_options = ["as-built", "stress relieved", "HIP", "aged", "solution aged"]
    subprocess_options = ["L-PBF", "E-PBF", "L-DED", "WAAM"]
    grain_options = ["cellular", "columnar", "equiaxed", "mixed", "martensitic", "dendritic"]
    for alloy in alloys:
        props = THERMAL_DB[alloy]
        for j in range(n_per_alloy):
            sub = rng.choice(subprocess_options, p=[0.62, 0.08, 0.2, 0.10])
            power = float(rng.uniform(120, 420) if sub == "L-PBF" else rng.uniform(500, 1800))
            speed = float(rng.uniform(600, 1600) if sub == "L-PBF" else rng.uniform(5, 25))
            hatch = float(rng.uniform(70, 140) if sub == "L-PBF" else rng.uniform(500, 1800))
            layer = float(rng.choice([20, 30, 40, 50, 60, 80]))
            ht = str(rng.choice(ht_options))
            rel_density = float(np.clip(rng.normal(99.1, 0.55), 96.0, 99.95))
            porosity = float(max(0.02, 100.0 - rel_density + rng.normal(0, 0.05)))
            grain_size = float(np.clip(rng.normal(28, 12), 3, 90))
            texture = float(np.clip(rng.normal(3.0, 1.2), 1.0, 8.5))
            ved = power / (speed * (hatch / 1000) * (layer / 1000)) if sub == "L-PBF" else np.nan
            base_strength = {
                "stainless steel": 520,
                "precipitation hardened steel": 920,
                "titanium alloy": 980,
                "nickel superalloy": 760,
                "aluminum alloy": 285,
                "cobalt alloy": 680,
                "maraging steel": 1050,
                "copper alloy": 310,
                "tool steel": 980,
            }.get(props["alloy_family"], 500)
            ht_bonus = {"as-built": 0, "stress relieved": 15, "HIP": 30, "aged": 120, "solution aged": 90}.get(ht, 0)
            density_bonus = (rel_density - 98.0) * 45
            porosity_penalty = porosity * 65
            grain_bonus = max(0, (35 - grain_size)) * 2.2
            y_s = base_strength + ht_bonus + density_bonus - porosity_penalty + grain_bonus + rng.normal(0, 45)
            uts = y_s + rng.uniform(80, 220)
            elong = np.clip(18 + (rel_density - 98.5) * 4 - porosity * 6 - (base_strength - 600) / 180 + rng.normal(0, 2), 1.5, 45)
            hardness = np.clip(y_s / 3.1 + rng.normal(0, 20), 70, 620)
            modulus = {"Fe": 200, "Ti": 115, "Ni": 205, "Al": 70, "Co": 220, "Cu": 120}.get(props["base_element"], 160) + rng.normal(0, 7)
            roughness = np.clip(10 + layer / 2 + rng.normal(0, 6), 4, 90)
            row = dict(
                sample_id=f"demo_{alloy.replace(' ', '_')}_{j:03d}",
                source_doi="demo_synthetic_literature_style",
                alloy=alloy,
                AM_process="PBF" if "PBF" in sub else "DED",
                AM_subprocess=sub,
                machine=str(rng.choice(["EOS M290", "Renishaw AM250", "Concept Laser M2", "Trumpf", "Optomec", "Sciaky", "WAAM lab"])),
                laser_power_W=round(power, 2),
                scan_speed_mm_s=round(speed, 2),
                hatch_spacing_um=round(hatch, 2),
                layer_thickness_um=round(layer, 2),
                beam_diameter_um=round(float(rng.uniform(60, 120)), 2),
                build_orientation=str(rng.choice(["vertical", "horizontal", "45deg"])),
                specimen_orientation=str(rng.choice(["XY", "XZ", "Z"])),
                heat_treatment=ht,
                relative_density_percent=round(rel_density, 3),
                porosity_percent=round(porosity, 3),
                grain_size_um=round(grain_size, 3),
                grain_morphology=str(rng.choice(grain_options)),
                texture_intensity_MRD=round(texture, 3),
                data_quality=str(rng.choice(["A", "B", "C"], p=[0.45, 0.4, 0.15])),
                yield_strength_MPa=round(float(y_s), 2),
                UTS_MPa=round(float(uts), 2),
                elongation_percent=round(float(elong), 2),
                hardness_HV=round(float(hardness), 2),
                elastic_modulus_GPa=round(float(modulus), 2),
                surface_roughness_Ra_um=round(float(roughness), 2),
            )
            row.update(props)
            rows.append(row)
    return add_physics_features(pd.DataFrame(rows))


def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ALL_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out


def load_uploaded_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)


def parse_pasted_csv(text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(text.strip()))


def model_feature_columns(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    exclude = set(TARGETS + ["sample_id", "source_doi"])
    features = [c for c in df.columns if c not in exclude and c != target]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in features if c not in numeric]
    return numeric, categorical


def build_model(df: pd.DataFrame, target: str, model_name: str = "Random Forest"):
    clean = df.dropna(subset=[target]).copy()
    if len(clean) < 5:
        raise ValueError(f"Need at least 5 rows with {target} to train. Current rows: {len(clean)}")
    numeric, categorical = model_feature_columns(clean, target)
    X = clean[numeric + categorical]
    y = pd.to_numeric(clean[target], errors="coerce")
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask]
    if model_name == "Gradient Boosting":
        reg = GradientBoostingRegressor(random_state=42)
    else:
        reg = RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=2)
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ],
        remainder="drop",
    )
    pipe = Pipeline([("pre", pre), ("model", reg)])
    pipe.fit(X, y)
    return pipe, numeric + categorical, clean


def evaluate_model(df: pd.DataFrame, target: str, model_name: str = "Random Forest") -> pd.DataFrame:
    clean = df.dropna(subset=[target]).copy()
    if len(clean) < 8:
        raise ValueError("Need at least 8 rows with target values for benchmark.")
    pipe, features, _ = build_model(clean, target, model_name)
    X = clean[features]
    y = pd.to_numeric(clean[target], errors="coerce")
    k = min(5, len(clean))
    cv = KFold(n_splits=k, shuffle=True, random_state=42)
    r2s = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
    maes = -cross_val_score(pipe, X, y, cv=cv, scoring="neg_mean_absolute_error")
    return pd.DataFrame([{"model": model_name, "rows": len(clean), "cv_folds": k, "mean_R2": round(float(np.nanmean(r2s)), 3), "mean_MAE": round(float(np.nanmean(maes)), 3)}])


def out_of_domain_warnings(train: pd.DataFrame, query: pd.DataFrame) -> List[str]:
    warnings = []
    if query.empty:
        return warnings
    q = query.iloc[0]
    for cat in ["alloy", "AM_subprocess", "heat_treatment", "grain_morphology", "matrix_phase"]:
        if cat in train.columns and cat in query.columns:
            val = str(q.get(cat, ""))
            if val and val != "nan" and val not in set(train[cat].astype(str).dropna()):
                warnings.append(f"{cat}='{val}' was not seen in training data.")
    for num in ["laser_power_W", "scan_speed_mm_s", "hatch_spacing_um", "layer_thickness_um", "volumetric_energy_density_J_mm3", "porosity_percent", "grain_size_um"]:
        if num in train.columns and num in query.columns:
            val = pd.to_numeric(pd.Series([q.get(num)]), errors="coerce").iloc[0]
            if pd.notna(val):
                lo, hi = pd.to_numeric(train[num], errors="coerce").quantile([0.01, 0.99])
                if pd.notna(lo) and pd.notna(hi) and (val < lo or val > hi):
                    warnings.append(f"{num}={val:.3g} is outside the training 1%-99% range [{lo:.3g}, {hi:.3g}].")
    if not warnings:
        warnings.append("No major out-of-domain warning detected, but this is still research guidance only.")
    return warnings


def nearest_cases(train: pd.DataFrame, query: pd.DataFrame, target: str, k: int = 5) -> pd.DataFrame:
    if train.empty or query.empty:
        return pd.DataFrame()
    numeric = [c for c in ["laser_power_W", "scan_speed_mm_s", "hatch_spacing_um", "layer_thickness_um", "relative_density_percent", "porosity_percent", "grain_size_um", "volumetric_energy_density_J_mm3"] if c in train.columns and c in query.columns]
    if not numeric:
        return pd.DataFrame()
    train_num = train[numeric].apply(pd.to_numeric, errors="coerce")
    query_num = query[numeric].apply(pd.to_numeric, errors="coerce")
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(train_num)
    q = imp.transform(query_num)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    qs = scaler.transform(q)
    nn = NearestNeighbors(n_neighbors=min(k, len(train)), metric="euclidean")
    nn.fit(Xs)
    dist, idx = nn.kneighbors(qs)
    cols = [c for c in ["sample_id", "source_doi", "alloy", "AM_subprocess", "heat_treatment", "laser_power_W", "scan_speed_mm_s", "hatch_spacing_um", "layer_thickness_um", "relative_density_percent", "porosity_percent", "grain_morphology", "matrix_phase", target] if c in train.columns]
    out = train.iloc[idx[0]][cols].copy()
    out.insert(0, "distance", np.round(dist[0], 3))
    return out


def csv_download_button(df: pd.DataFrame, label: str, file_name: str):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=file_name, mime="text/csv")


def manual_row_form(prefix: str = "manual") -> Dict[str, object]:
    st.markdown("### Manual training sample entry")
    st.caption("Fill one sample/experiment row. Add target property values if this row should be used for training.")
    c1, c2, c3 = st.columns(3)
    with c1:
        sample_id = st.text_input("sample_id", value=f"manual_{len(st.session_state.get('manual_rows', []))+1:03d}", help=PARAM_HELP["sample_id"], key=f"{prefix}_sample_id")
        source_doi = st.text_input("source_doi / lab ID", value="manual", help=PARAM_HELP["source_doi"], key=f"{prefix}_source")
        alloy = st.selectbox("alloy", list(THERMAL_DB.keys()) + ["Other"], help=PARAM_HELP["alloy"], key=f"{prefix}_alloy")
        if alloy == "Other":
            alloy = st.text_input("custom alloy name", value="Custom alloy", key=f"{prefix}_custom_alloy")
        AM_subprocess = st.selectbox("AM_subprocess", ["L-PBF", "E-PBF", "L-DED", "WAAM", "Binder Jet", "Other"], help=PARAM_HELP["AM_subprocess"], key=f"{prefix}_subprocess")
        AM_process = "PBF" if "PBF" in AM_subprocess else "DED" if AM_subprocess in ["L-DED", "WAAM"] else AM_subprocess
        machine = st.text_input("machine", value="unknown", key=f"{prefix}_machine")
    with c2:
        laser_power_W = st.number_input("laser_power_W", min_value=0.0, value=250.0, help=PARAM_HELP["laser_power_W"], key=f"{prefix}_P")
        scan_speed_mm_s = st.number_input("scan_speed_mm_s", min_value=0.0, value=900.0, help=PARAM_HELP["scan_speed_mm_s"], key=f"{prefix}_v")
        hatch_spacing_um = st.number_input("hatch_spacing_um", min_value=0.0, value=100.0, help=PARAM_HELP["hatch_spacing_um"], key=f"{prefix}_h")
        layer_thickness_um = st.number_input("layer_thickness_um", min_value=0.0, value=30.0, help=PARAM_HELP["layer_thickness_um"], key=f"{prefix}_t")
        beam_diameter_um = st.number_input("beam_diameter_um", min_value=0.0, value=80.0, help=PARAM_HELP["beam_diameter_um"], key=f"{prefix}_bd")
        heat_treatment = st.selectbox("heat_treatment", ["as-built", "stress relieved", "annealed", "HIP", "aged", "solution aged", "unknown"], help=PARAM_HELP["heat_treatment"], key=f"{prefix}_ht")
    with c3:
        relative_density_percent = st.number_input("relative_density_percent", min_value=0.0, max_value=100.0, value=99.0, help=PARAM_HELP["relative_density_percent"], key=f"{prefix}_rd")
        porosity_percent = st.number_input("porosity_percent", min_value=0.0, max_value=100.0, value=1.0, help=PARAM_HELP["porosity_percent"], key=f"{prefix}_por")
        grain_size_um = st.number_input("grain_size_um", min_value=0.0, value=25.0, help=PARAM_HELP["grain_size_um"], key=f"{prefix}_grain")
        texture_intensity_MRD = st.number_input("texture_intensity_MRD", min_value=0.0, value=2.5, help=PARAM_HELP["texture_intensity_MRD"], key=f"{prefix}_tex")
        grain_morphology = st.selectbox("grain_morphology", ["unknown", "cellular", "columnar", "equiaxed", "mixed", "martensitic", "dendritic"], key=f"{prefix}_gm")
        data_quality = st.selectbox("data_quality", ["A", "B", "C", "D"], key=f"{prefix}_dq")

    st.markdown("### Target values for training")
    tcols = st.columns(4)
    target_vals = {}
    defaults = {
        "yield_strength_MPa": 650.0,
        "UTS_MPa": 800.0,
        "elongation_percent": 12.0,
        "hardness_HV": 250.0,
        "elastic_modulus_GPa": 160.0,
        "surface_roughness_Ra_um": 25.0,
        "relative_density_percent": relative_density_percent,
    }
    for i, target in enumerate(TARGETS):
        with tcols[i % 4]:
            use_val = st.checkbox(f"include {target}", value=target in ["yield_strength_MPa", "UTS_MPa"], key=f"{prefix}_use_{target}")
            if use_val:
                target_vals[target] = st.number_input(target, value=float(defaults[target]), key=f"{prefix}_{target}")
            else:
                target_vals[target] = np.nan

    row = dict(
        sample_id=sample_id,
        source_doi=source_doi,
        alloy=alloy,
        AM_process=AM_process,
        AM_subprocess=AM_subprocess,
        machine=machine,
        laser_power_W=laser_power_W,
        scan_speed_mm_s=scan_speed_mm_s,
        hatch_spacing_um=hatch_spacing_um,
        layer_thickness_um=layer_thickness_um,
        beam_diameter_um=beam_diameter_um,
        build_orientation="unknown",
        specimen_orientation="unknown",
        heat_treatment=heat_treatment,
        relative_density_percent=relative_density_percent,
        porosity_percent=porosity_percent,
        grain_size_um=grain_size_um,
        grain_morphology=grain_morphology,
        texture_intensity_MRD=texture_intensity_MRD,
        data_quality=data_quality,
    )
    row.update(target_vals)
    props = THERMAL_DB.get(alloy, {})
    row.update({k: v for k, v in props.items() if k not in row or pd.isna(row.get(k))})
    return row


def show_parameter_dictionary():
    st.header("Data dictionary: what each input means")
    rows = []
    for col in ALL_COLUMNS:
        rows.append({"column": col, "meaning": PARAM_HELP.get(col, "Training/prediction field used by the model or metadata for traceability.")})
    st.table(pd.DataFrame(rows))

# Session state
if "manual_rows" not in st.session_state:
    st.session_state.manual_rows = []

# Sidebar
st.sidebar.title("Data source")
uploaded = st.sidebar.file_uploader("Upload your training CSV", type=["csv"])
auto_fill = st.sidebar.checkbox("Auto-fill missing alloy thermal properties", value=True)
target = st.sidebar.selectbox("Target property", TARGETS)
page = st.sidebar.radio("Page", [
    "1 Overview",
    "2 Data cockpit",
    "3 Alloy passport",
    "4 Manual training builder",
    "5 Train benchmark",
    "6 Predict / user data",
    "7 Microstructure + phases",
    "8 Data dictionary",
])

# Load base dataset
if uploaded is not None:
    base_df = load_uploaded_csv(uploaded)
else:
    base_df = make_demo_dataset(n_per_alloy=30)

manual_df = pd.DataFrame(st.session_state.manual_rows)
if not manual_df.empty:
    base_df = pd.concat([base_df, manual_df], ignore_index=True, sort=False)

base_df = ensure_cols(base_df)
if auto_fill:
    base_df = auto_fill_thermal(base_df)
base_df = add_physics_features(base_df)

st.title("OpenMetalAM-AI")
st.caption("Stable app with CSV upload, manual training data entry, physics-aware features, microstructure/phase fields, and user prediction.")

if page == "1 Overview":
    st.header("Dataset coverage snapshot")
    overview = pd.DataFrame([
        {"item": "Rows", "value": len(base_df)},
        {"item": "Alloys", "value": base_df["alloy"].nunique(dropna=True)},
        {"item": "AM subprocesses", "value": base_df["AM_subprocess"].nunique(dropna=True)},
        {"item": "Heat treatments", "value": base_df["heat_treatment"].nunique(dropna=True)},
        {"item": "Manual rows added this session", "value": len(st.session_state.manual_rows)},
    ])
    st.table(overview)
    st.subheader("Why this app exists")
    st.write("This app lets a user train AM metallurgy models from uploaded CSV data, bundled demo data, and rows entered manually by hand. It is designed for process → microstructure/phases → property prediction.")
    st.subheader("Preview")
    preview_cols = [c for c in ["sample_id", "alloy", "AM_subprocess", "heat_treatment", "laser_power_W", "scan_speed_mm_s", "hatch_spacing_um", "layer_thickness_um", "volumetric_energy_density_J_mm3", "grain_morphology", "matrix_phase", target] if c in base_df.columns]
    st.dataframe(base_df[preview_cols].head(30), width="stretch")

elif page == "2 Data cockpit":
    st.header("Data cockpit")
    c1, c2, c3 = st.columns(3)
    with c1:
        alloys = st.multiselect("Filter alloys", sorted(base_df["alloy"].dropna().astype(str).unique()), default=[])
    with c2:
        subs = st.multiselect("Filter AM subprocess", sorted(base_df["AM_subprocess"].dropna().astype(str).unique()), default=[])
    with c3:
        hts = st.multiselect("Filter heat treatment", sorted(base_df["heat_treatment"].dropna().astype(str).unique()), default=[])
    df = base_df.copy()
    if alloys:
        df = df[df["alloy"].astype(str).isin(alloys)]
    if subs:
        df = df[df["AM_subprocess"].astype(str).isin(subs)]
    if hts:
        df = df[df["heat_treatment"].astype(str).isin(hts)]
    cols = [c for c in ["sample_id", "source_doi", "alloy", "alloy_family", "AM_subprocess", "heat_treatment", "laser_power_W", "scan_speed_mm_s", "hatch_spacing_um", "layer_thickness_um", "linear_energy_density_J_mm", "volumetric_energy_density_J_mm3", "relative_density_percent", "porosity_percent", "grain_size_um", "grain_morphology", "matrix_phase", "secondary_phases", target] if c in df.columns]
    st.write(f"Rows after filters: {len(df)}")
    st.dataframe(df[cols], width="stretch")
    csv_download_button(df[cols], "Download filtered data", "filtered_openmetalam_data.csv")

elif page == "3 Alloy passport":
    st.header("Alloy passport")
    alloy = st.selectbox("Choose alloy", sorted(base_df["alloy"].dropna().astype(str).unique()))
    dfa = base_df[base_df["alloy"].astype(str) == alloy]
    info_cols = ["alloy", "alloy_family", "base_element", "density_g_cm3", "melting_temp_C", "thermal_conductivity_W_mK", "specific_heat_J_kgK", "CTE_1e6_K", "matrix_phase", "secondary_phases"]
    st.subheader("Material identity")
    st.table(dfa[info_cols].head(1).T.rename(columns={dfa[info_cols].head(1).index[0]: "value"}) if len(dfa) else pd.DataFrame())
    st.subheader("Property summary")
    props = [p for p in TARGETS if p in dfa.columns]
    st.table(dfa[props].describe().T.round(3))
    st.subheader("Representative rows")
    cols = [c for c in ["sample_id", "AM_subprocess", "heat_treatment", "laser_power_W", "scan_speed_mm_s", "relative_density_percent", "porosity_percent", "grain_morphology", "matrix_phase", target] if c in dfa.columns]
    st.dataframe(dfa[cols].head(50), width="stretch")

elif page == "4 Manual training builder":
    st.header("Manual training data builder")
    st.write("Use this page when you do not have a CSV yet. Add rows by hand, then train the model or download a CSV template.")
    with st.form("manual_row_form"):
        row = manual_row_form(prefix="manual_form")
        submitted = st.form_submit_button("Add this sample to training data")
        if submitted:
            st.session_state.manual_rows.append(row)
            st.success("Manual row added. It is now included in the training dataset for this session.")
    st.subheader("Manual rows in this session")
    if st.session_state.manual_rows:
        mdf = add_physics_features(auto_fill_thermal(ensure_cols(pd.DataFrame(st.session_state.manual_rows))))
        st.dataframe(mdf, width="stretch")
        csv_download_button(mdf, "Download manual rows as CSV", "manual_training_rows.csv")
        if st.button("Clear manual rows"):
            st.session_state.manual_rows = []
            st.rerun()
    else:
        st.info("No manual rows added yet.")
    st.subheader("Bulk paste CSV")
    st.write("Paste rows with headers. This is useful if you copied data from Excel or a paper table.")
    example = pd.DataFrame([ensure_cols(pd.DataFrame([manual_row_form(prefix="hidden_example")])).iloc[0].to_dict()]).head(0)
    pasted = st.text_area("Paste CSV text here", height=180, placeholder="sample_id,alloy,AM_subprocess,laser_power_W,scan_speed_mm_s,...")
    if st.button("Add pasted CSV rows"):
        try:
            parsed = parse_pasted_csv(pasted)
            st.session_state.manual_rows.extend(parsed.to_dict(orient="records"))
            st.success(f"Added {len(parsed)} pasted rows.")
            st.rerun()
        except Exception as e:
            st.error(f"Could not parse pasted CSV: {e}")
    st.subheader("Empty template")
    template = pd.DataFrame(columns=ALL_COLUMNS)
    st.dataframe(template, width="stretch")
    csv_download_button(template, "Download empty training CSV template", "openmetalam_training_template.csv")

elif page == "5 Train benchmark":
    st.header("Train benchmark")
    st.write(f"Target: `{target}`")
    results = []
    for model_name in ["Random Forest", "Gradient Boosting"]:
        try:
            results.append(evaluate_model(base_df, target, model_name))
        except Exception as e:
            st.warning(f"{model_name} failed: {e}")
    if results:
        st.table(pd.concat(results, ignore_index=True))
    st.subheader("Training rows used")
    usable = base_df.dropna(subset=[target])
    st.write(f"Rows with target `{target}`: {len(usable)}")
    cols = [c for c in ["sample_id", "alloy", "AM_subprocess", "heat_treatment", target] if c in usable.columns]
    st.dataframe(usable[cols].head(100), width="stretch")

elif page == "6 Predict / user data":
    st.header("Predict / user data")
    st.write("Train on the current dataset, then predict one manual query or an uploaded query CSV.")
    model_name = st.selectbox("Model", ["Random Forest", "Gradient Boosting"])
    mode = st.radio("Prediction input mode", ["Manual single query", "Upload query CSV"])
    if mode == "Manual single query":
        qrow = manual_row_form(prefix="query")
        query_df = pd.DataFrame([qrow])
    else:
        qfile = st.file_uploader("Upload query CSV. It does not need target columns.", type=["csv"], key="query_csv")
        query_df = load_uploaded_csv(qfile) if qfile else pd.DataFrame()
    if st.button("Train and predict"):
        try:
            train_df = base_df.dropna(subset=[target]).copy()
            train_df = add_physics_features(auto_fill_thermal(ensure_cols(train_df)))
            query_df = add_physics_features(auto_fill_thermal(ensure_cols(query_df)))
            pipe, features, clean = build_model(train_df, target, model_name)
            preds = pipe.predict(query_df[features])
            out = query_df[[c for c in ["sample_id", "alloy", "AM_subprocess", "heat_treatment", "laser_power_W", "scan_speed_mm_s", "hatch_spacing_um", "layer_thickness_um", "grain_morphology", "matrix_phase"] if c in query_df.columns]].copy()
            out[f"predicted_{target}"] = np.round(preds, 3)
            st.subheader("Prediction")
            st.dataframe(out, width="stretch")
            st.subheader("Warnings")
            for w in out_of_domain_warnings(clean, query_df):
                st.write(f"- {w}")
            st.subheader("Nearest training/literature-style cases")
            st.dataframe(nearest_cases(clean, query_df, target, k=7), width="stretch")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

elif page == "7 Microstructure + phases":
    st.header("Microstructure + phases")
    st.write("This page shows the V2/V3 extension: microstructure, crystallography/texture, and phase descriptors used alongside process parameters.")
    cols = [c for c in ["sample_id", "alloy", "AM_subprocess", "heat_treatment", "relative_density_percent", "porosity_percent", "grain_size_um", "grain_morphology", "texture_intensity_MRD", "matrix_phase", "secondary_phases", target] if c in base_df.columns]
    st.dataframe(base_df[cols].head(150), width="stretch")
    st.subheader("Group summary")
    group = base_df.groupby(["alloy", "grain_morphology", "matrix_phase"], dropna=False)[target].agg(["count", "mean", "std"]).reset_index().sort_values("count", ascending=False).head(50)
    st.dataframe(group.round(3), width="stretch")

elif page == "8 Data dictionary":
    show_parameter_dictionary()

st.markdown("---")
st.caption("Research guidance only. Not a certified property database, qualification tool, or replacement for experiments.")
