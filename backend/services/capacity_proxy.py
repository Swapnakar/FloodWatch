"""
Drain capacity PROXY from pipe diameter (Manning's equation, full-bore circular pipe).

    Q = (1 / n) * A * R^(2/3) * S^(1/2)

    A = pi * D^2 / 4     flow area of a full circular pipe (m^2)
    R = D / 4            hydraulic radius of a full circular pipe (m)
    n = 0.013            Manning roughness, concrete/RCC pipe (standard textbook value)
    S = 0.001            ASSUMED nominal bed slope (1 in 1000). PLACEHOLDER: the KMC
                         maps give no pipe slopes or invert levels usable per segment.

The result is an ESTIMATE from pipe diameter only. It is NOT KMC's measured or
design flow capacity. It ignores the true slope, silt, blockage, surcharge,
partial flow, pipe condition and downstream constraints. Because S is one
constant, the value ranks pipes by size (Q grows with D^(8/3)) rather than
giving their real capacity. Every value is exposed as
`drain_capacity_estimated_m3s` together with CAPACITY_DISCLAIMER.
"""

import math
from typing import Dict, Optional

MANNING_N = 0.013
ASSUMED_SLOPE = 0.001
MIN_DIAMETER_MM = 100.0
MAX_DIAMETER_MM = 3000.0

CAPACITY_FIELD = "drain_capacity_estimated_m3s"
CAPACITY_METHOD = "manning_full_bore_circular"
CAPACITY_DISCLAIMER = (
    "ESTIMATED via Manning's equation from pipe diameter only "
    f"(n={MANNING_N}, assumed slope={ASSUMED_SLOPE}); not measured KMC flow capacity."
)


def manning_full_pipe_capacity_m3s(
    diameter_mm: Optional[float],
    slope: float = ASSUMED_SLOPE,
    n: float = MANNING_N,
) -> Optional[float]:
    """Full-bore capacity (m^3/s) of a circular pipe. None if diameter is unknown or implausible."""
    if diameter_mm is None:
        return None
    try:
        d_mm = float(diameter_mm)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(d_mm) or not (MIN_DIAMETER_MM <= d_mm <= MAX_DIAMETER_MM):
        return None
    if not (slope > 0 and n > 0):
        raise ValueError("slope and n must be positive")
    d = d_mm / 1000.0
    area = math.pi * d * d / 4.0
    hydraulic_radius = d / 4.0
    return (1.0 / n) * area * hydraulic_radius ** (2.0 / 3.0) * math.sqrt(slope)


def capacity_fields(diameter_mm: Optional[float], conduit_type: Optional[str] = "pipe") -> Dict:
    """Response fragment: the estimate plus its provenance. Never presented bare.

    Only circular pipes are estimated. Box sewers and drains without a diameter
    get None, not a made-up equivalent.
    """
    value = manning_full_pipe_capacity_m3s(diameter_mm) if conduit_type == "pipe" else None
    return {
        CAPACITY_FIELD: round(value, 4) if value is not None else None,
        "drain_capacity_method": CAPACITY_METHOD if value is not None else None,
        "drain_capacity_note": CAPACITY_DISCLAIMER if value is not None else
        "No estimate: diameter unknown or conduit is not a circular pipe.",
    }
