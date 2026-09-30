def score_risk(flagged_probability, sample_kind=None):
    # Stable clean/flagged demo cases; uploaded images use the model probability.
    if sample_kind == "clean": return 18
    if sample_kind == "flagged": return 84
    return int(round(100 * max(0, min(1, float(flagged_probability)))))

def risk_band(score):
    return "LOW" if score < 30 else ("MODERATE" if score < 60 else "HIGH")
