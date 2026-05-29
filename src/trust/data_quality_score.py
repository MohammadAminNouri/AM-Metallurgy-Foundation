def score_row(row: dict) -> str:
    """Simple rule-based quality score for literature/user data."""
    required_core = ["alloy", "AM_process"]
    process_cols = ["laser_power_W", "scan_speed_mm_s", "layer_thickness_um"]
    labels = ["yield_strength_MPa", "UTS_MPa", "elongation_percent", "hardness_HV"]
    has_core = all(row.get(c) not in [None, ""] for c in required_core)
    process_count = sum(row.get(c) not in [None, ""] for c in process_cols)
    label_count = sum(row.get(c) not in [None, ""] for c in labels)
    extraction = str(row.get("extraction_method", "")).lower()
    if has_core and process_count >= 3 and label_count >= 1 and extraction in ["table", "direct_table", "user"]:
        return "A"
    if has_core and process_count >= 2 and label_count >= 1:
        return "B"
    if has_core and label_count >= 1:
        return "C"
    if has_core:
        return "D"
    return "E"
