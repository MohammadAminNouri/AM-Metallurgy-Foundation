# Elevated app notes

This app was upgraded after the first scaffold looked empty and hard to interpret.

## Major changes

1. Added 240-row bundled demo dataset for UI and pipeline testing.
2. Added alloy passport view so users can understand alloy family, thermal properties, matrix phase, and microstructure expectation.
3. Added parameter dictionary explaining process, structure, crystallography, phase, and property columns.
4. Added prediction page with manual form and query CSV upload.
5. Added trust warnings and nearest similar cases.
6. Added microstructure/phase page to expose the V2/V3 concept directly.
7. Added templates for user training and prediction CSVs.

## What still requires real data work

- Import and license-check public datasets.
- Curate real literature rows with source DOI/table/figure tracking.
- Add SHAP plots.
- Add source-paper split, alloy-family split, and leave-one-alloy-out validation.
- Add true image/EBSD/XRD modules.
