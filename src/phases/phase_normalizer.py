PHASE_ALIASES = {
    "alpha_prime_martensite": ["alpha prime", "α'", "alpha-prime", "martensite"],
    "austenite": ["austenite", "gamma", "γ"],
    "ferrite": ["ferrite", "alpha", "α"],
    "beta": ["beta", "β"],
    "gamma_prime": ["gamma prime", "γ'"],
    "gamma_double_prime": ["gamma double prime", "γ\"", "gamma\""],
    "laves": ["laves"],
    "carbide": ["carbide", "mc", "m23c6"],
}


def normalize_phase_text(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    t = text.lower()
    phases = []
    for canonical, aliases in PHASE_ALIASES.items():
        if any(a.lower() in t for a in aliases):
            phases.append(canonical)
    return sorted(set(phases))
