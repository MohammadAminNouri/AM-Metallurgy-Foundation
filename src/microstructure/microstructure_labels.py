MICROSTRUCTURE_LABELS = [
    "equiaxed", "columnar", "mixed", "cellular", "dendritic", "martensitic",
    "lamellar", "bimodal", "fine_grains", "coarse_grains"
]


def normalize_grain_morphology(value: str) -> str:
    if not isinstance(value, str):
        return "unknown"
    v = value.lower().strip()
    if "column" in v:
        return "columnar"
    if "equiax" in v:
        return "equiaxed"
    if "cell" in v:
        return "cellular"
    if "dend" in v:
        return "dendritic"
    if "martens" in v or "alpha prime" in v or "α'" in v:
        return "martensitic"
    if "lamellar" in v:
        return "lamellar"
    if "bimodal" in v:
        return "bimodal"
    return v or "unknown"
