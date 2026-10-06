"""
Safe-route service: score, compare and recommend the lower-flood-risk route.

Per the plan (section 22-24):
  - Route score = weighted mean of per-segment flood_probability along the route.
    Configurable weights for MAX, P90, MEAN (the plan's weighted formula).
  - Decision logic: pick the route with the lowest score, but NEVER call any
    route "100% safe" or "no risk". If all routes are risky, say so explicitly.
  - Every response names its model version and the provenance of the prediction.
"""

from typing import Dict, List, Optional

import numpy as np

# Section 22's weighted formula: score = w_max*max + w_p90*p90 + w_mean*mean.
W_MAX = 0.30
W_P90 = 0.35
W_MEAN = 0.35

ALL_RISKY_THRESHOLD = 0.50  # above this, ALL routes get a warning


def route_score(probabilities: List[Optional[float]]) -> Optional[float]:
    """Composite risk score for a route from its segment probabilities.

    Returns None when no segment has a probability (no model / no features).
    """
    vals = [p for p in probabilities if p is not None]
    if not vals:
        return None
    a = np.array(vals)
    return round(float(W_MAX * a.max() + W_P90 * np.percentile(a, 90) + W_MEAN * a.mean()), 4)


def risk_level_for_route(score: Optional[float]) -> Optional[str]:
    if score is None:
        return None
    if score >= 0.70:
        return "HIGH"
    if score >= 0.45:
        return "ELEVATED"
    if score >= 0.25:
        return "MODERATE"
    return "LOW"


def pick_recommendation(routes: List[Dict]) -> Dict:
    """Choose the lower-risk route and frame the recommendation honestly.

    Returns {"recommended_index", "recommendation", "all_risky", "comparison"}.
    NEVER uses the word "safe" as a claim of safety.
    """
    scored = [(i, r["route_score"]) for i, r in enumerate(routes) if r.get("route_score") is not None]
    if not scored:
        return {
            "recommended_index": None,
            "recommendation": "Unable to assess flood risk for these routes: insufficient feature coverage.",
            "all_risky": None,
            "comparison": None,
        }
    scored.sort(key=lambda t: t[1])
    best_i, best_score = scored[0]
    all_risky = all(s >= ALL_RISKY_THRESHOLD for _, s in scored)
    if all_risky:
        rec = ("All available routes have elevated flood risk. "
               f"The least affected route is route {best_i + 1}, but exercise caution and monitor conditions.")
    elif len(scored) > 1 and scored[0][1] < scored[1][1]:
        diff = scored[1][1] - scored[0][1]
        rec = (f"Route {best_i + 1} has lower flood risk (score {best_score:.2f} vs {scored[1][1]:.2f}, "
               f"difference {diff:.2f}). Neither route is guaranteed flood-free.")
    elif len(scored) > 1:
        rec = (f"Routes have similar flood risk (scores {scored[0][1]:.2f} and {scored[1][1]:.2f}). "
               "Choose based on other preferences.")
    else:
        rec = (f"One route assessed with flood risk score {best_score:.2f}. "
               "This is a relative measure, not a safety guarantee.")

    comparison = [{"route_index": i, "score": s, "risk_level": risk_level_for_route(s)}
                  for i, s in scored]
    return {
        "recommended_index": best_i,
        "recommendation": rec,
        "all_risky": all_risky,
        "comparison": comparison,
    }
