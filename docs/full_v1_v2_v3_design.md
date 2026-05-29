# OpenMetalAM-AI: Full V1 + V2 + V3 Design

## Core idea
A repository for metal additive manufacturing that joins process, microstructure, crystallography, phases, and properties.

## Layered pipeline

1. Literature/user data ingestion
2. Cleaning and unit standardization
3. Physics-aware feature engineering
4. Microstructure/crystallography/phase enrichment
5. Mechanical-property prediction
6. Explainability
7. Warnings and nearest literature cases
8. App interface for users

## Model tasks

### Task A: Process -> Property
Predict YS, UTS, elongation, hardness, elastic modulus, roughness, and relative density.

### Task B: Process -> Microstructure
Predict grain morphology, porosity class, melt pool geometry, defect risk, and phase labels when enough data exists.

### Task C: Microstructure/Phase -> Property
Use grain size, phase fraction, texture, porosity, and precipitates to predict properties.

### Task D: Full chain
Process + alloy + heat treatment + microstructure + crystallography + phases -> properties.

## User-upload support
The app accepts a user CSV for training and a user CSV for prediction. This allows users to train on their own lab data or combine lab data with public/literature data.

## Data-quality labels
A: Direct table value with full process parameters.
B: Direct table value with partial process parameters.
C: Digitized from figure or qualitative label.
D: Inferred or incomplete.
E: Archive only, not recommended for model training.

## Public integrations
- MechProNet-style mechanical-property data
- MeltpoolNet melt-pool regression/classification data
- NIST AM-Bench process/microstructure measurements
- Public melt-pool image compilations
- User-provided data

## Limitations
The repository is a research framework. It does not certify parts, replace experiments, or guarantee industrial qualification.
