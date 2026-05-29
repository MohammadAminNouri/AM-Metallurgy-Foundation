# OpenMetalAM-AI

A metallurgy-aware Streamlit and Python scaffold for metal additive manufacturing (AM) data, process–microstructure–phase–property modeling, and user-uploaded datasets.

## What is inside now

- A stronger Streamlit app with pages for overview, data cockpit, alloy passport, benchmarking, prediction, microstructure/phases, data dictionary, and roadmap.
- A bundled **literature-style demo dataset** with 240 rows across common metal AM alloys. This is for testing the pipeline and UI, not for certified scientific claims.
- Alloy reference table with thermal/material descriptors.
- Parameter dictionary explaining what every key column means.
- User CSV templates for training and prediction.
- Feature generation for line energy, volumetric energy density, beam power density, thermal diffusivity, density/porosity proxies.
- Random Forest and Gradient Boosting training.
- Nearest-case retrieval and out-of-domain warnings.

## Run locally

```bash
pip install -r requirements.txt
streamlit run src/app/streamlit_app.py
```

## Streamlit Cloud main file

```text
src/app/streamlit_app.py
```

## Important limitation

The bundled demo dataset is synthetic/literature-style starter data. Replace it with real curated literature data, MeltpoolNet/NIST/AM-Bench imports, or your own lab data before making scientific claims.

## Minimum user CSV columns

```text
alloy, AM_subprocess, laser_power_W, scan_speed_mm_s, hatch_spacing_um, layer_thickness_um, heat_treatment
```

Better V3 columns:

```text
relative_density_percent, porosity_percent, grain_size_um, grain_morphology, texture_intensity_MRD, matrix_phase, secondary_phases
```

Targets for training:

```text
yield_strength_MPa, UTS_MPa, elongation_percent, hardness_HV, elastic_modulus_GPa
```
