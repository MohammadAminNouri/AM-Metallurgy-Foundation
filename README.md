# OpenMetalAM-AI

A traceable, literature-mined, metallurgy-aware machine-learning framework for metal additive manufacturing.

## Scope
OpenMetalAM-AI links:

`process parameters -> thermal/material descriptors -> defects -> microstructure -> crystallography/phases -> mechanical properties`

The repository is designed for three use cases:

1. Build a public literature-mined metal AM dataset.
2. Predict mechanical properties from process/material/microstructure data.
3. Let users upload their own CSV data and train/predict with the same pipeline.

## Current integrated roadmap: V1 + V2 + V3

### V1: Mechanical-property predictor
- Literature-mined dataset schema
- Unit conversion and standardization
- Physics-aware feature engineering
- Random Forest / XGBoost / Gradient Boosting baselines
- SHAP explainability
- Nearest-literature-case retrieval
- Out-of-domain warnings
- Streamlit app

### V2: Microstructure, crystallography, and phases
- Microstructure table
- Crystallography table
- Phase table
- Qualitative and quantitative labels
- Data-quality scoring
- EBSD/XRD/SEM/XCT metadata support

### V3: Microstructure-aware property prediction
- Process -> microstructure models
- Microstructure/phase/crystallography -> property models
- Full process + structure + phase + property pipeline
- User-upload CSV support
- Model benchmark and data-quality dashboard

## Important limitation
This repo does not replace experiments or qualification. Predictions are based on public or user-provided data and should be treated as research guidance, not certified material-property values.

## Quick start

```bash
pip install -r requirements.txt
streamlit run src/app/streamlit_app.py
```

## User data
Users can upload their own CSV in the app. Required columns depend on the task, but the safest template is in:

`data/examples/user_input_template.csv`

## Data sources to integrate
This scaffold supports integration of:
- MechProNet-style mechanical-property datasets
- MeltpoolNet melt-pool geometry/defect datasets
- NIST AM-Bench data
- User-provided CSV data
- Literature-mined process/microstructure/property tables

Actual redistribution of third-party datasets must follow their licenses.
