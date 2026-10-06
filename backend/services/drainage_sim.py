"""
Drainage network rainfall simulation (scenario tool, not a forecast).

For a design storm (intensity mm/hr, duration min) estimate how loaded each
KMC sewer segment is:

  inflow   Q_in  = C * i * A_up * f_tc / 3.6e6        (Rational method, m^3/s)
  capacity Q_cap = Manning full-bore (capacity_proxy)   (m^3/s)
  load           = Q_in / Q_cap

  A_up   upstream contributing area: each pipe drains a strip of
         SERVICE_WIDTH_M along its length, accumulated downstream through the
         network graph (flow direction from the KMC flow arrows).
  C      runoff coefficient; rises with duration as ground saturates.
  f_tc   fraction of the upstream area that has reached the pipe: storms
         shorter than the time of concentration don't load the pipe fully.

Alert bands on load:  GREEN < 0.75 <= YELLOW < 1.0 <= ORANGE < 1.5 <= RED.

Assumptions are deliberate and returned with every result: constant slope
(the DEM is too coarse for pipe gradients), uniform service strip, no
silt/blockage, no pumping or tidal backwater. The KMC network extraction also
has gaps, so accumulation under-counts upstream area where pipes don't join.
"""

import json
import math
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from services.capacity_proxy import manning_full_pipe_capacity_m3s

DRAINAGE_GEOJSON = Path(__file__).resolve().parent.parent / "data" / "gis" / "drainage_network.geojson"

SERVICE_WIDTH_M = 40.0          # catchment strip drained per metre of pipe
FLOW_VELOCITY_MS = 0.9          # in-pipe velocity for time of concentration
INLET_TIME_MIN = 10.0           # overland flow time to the first inlet
C_MIN, C_MAX = 0.65, 0.90       # runoff coefficient, dry start -> saturated
C_SATURATION_MIN = 180.0        # minutes of rain to reach C_MAX
NODE_SNAP_M = 1.0

ALERT_BANDS = [(1.5, "RED"), (1.0, "ORANGE"), (0.75, "YELLOW"), (0.0, "GREEN")]
ALERT_MEANING = {
    "GREEN": "Within capacity",
    "YELLOW": "Near full (75-100% of capacity)",
    "ORANGE": "Surcharged (100-150%): water backing up into manholes",
    "RED": "Overflowing (>150%): street flooding likely",
}
ASSUMPTIONS = [
    "Capacity from pipe diameter via Manning (n=0.013, slope 1:1000); real slopes unknown.",
    f"Each pipe drains a {SERVICE_WIDTH_M:.0f} m wide strip; upstream area accumulated along KMC flow arrows.",
    "Runoff coefficient rises from 0.65 to 0.90 over 3 h of rain (ground saturation).",
    "No silt, blockage, pumping or tidal backwater modelled. Gaps in the digitised network reduce upstream area.",
]


def alert_for(load: Optional[float]) -> str:
    if load is None:
        return "UNKNOWN"
    for threshold, label in ALERT_BANDS:
        if load >= threshold:
            return label
    return "GREEN"


def _metres_per_deg(lat: float):
    return 111320.0 * math.cos(math.radians(lat)), 110540.0


class DrainageNetwork:
    """Pipe graph with static per-pipe hydraulics, built once."""

    def __init__(self, path: Path = DRAINAGE_GEOJSON):
        self.path = Path(path)
        with open(self.path) as f:
            raw = json.load(f).get("features", [])

        self.features: List[Dict] = []
        pipes = []  # (up_node, down_node, length_m, capacity_m3s|None)
        for feat in raw:
            coords = (feat.get("geometry") or {}).get("coordinates") or []
            if len(coords) < 2:
                continue
            props = feat.get("properties") or {}
            if props.get("flow_arrow") == "start":
                coords = coords[::-1]  # make coords run in the flow direction
            mx, my = _metres_per_deg(coords[0][1])
            length = sum(math.hypot((b[0] - a[0]) * mx, (b[1] - a[1]) * my)
                         for a, b in zip(coords, coords[1:]))
            cap = (manning_full_pipe_capacity_m3s(props.get("pipe_diameter_mm"))
                   if props.get("conduit_type", "pipe") == "pipe" else None)

            def node(pt):
                return (round(pt[0] * mx / NODE_SNAP_M), round(pt[1] * my / NODE_SNAP_M))

            pipes.append((node(coords[0]), node(coords[-1]), max(length, 1.0), cap))
            self.features.append({
                "coords": [[round(c[0], 6), round(c[1], 6)] for c in coords],
                "segment_id": props.get("segment_id"),
                "ward": props.get("ward"),
                "pipe_diameter_mm": props.get("pipe_diameter_mm"),
                "conduit_type": props.get("conduit_type"),
                "mid": coords[len(coords) // 2],
            })

        self.capacity = [p[3] for p in pipes]
        self.length = [p[2] for p in pipes]
        self.upstream_area_m2, self.flow_path_m = self._accumulate(pipes)
        self.index_by_segment = {f["segment_id"]: i for i, f in enumerate(self.features)}

    @staticmethod
    def _accumulate(pipes):
        """Route each pipe's local strip area downstream (topological order).
        Flow splits equally where a node has several outgoing pipes."""
        n = len(pipes)
        out_of = defaultdict(list)   # node -> pipes leaving it
        into = defaultdict(list)     # node -> pipes arriving at it
        for i, (u, d, _, _) in enumerate(pipes):
            out_of[u].append(i)
            into[d].append(i)

        indeg = [len(into[pipes[i][0]]) for i in range(n)]
        area = [pipes[i][2] * SERVICE_WIDTH_M for i in range(n)]
        path = [pipes[i][2] for i in range(n)]
        queue = deque(i for i in range(n) if indeg[i] == 0)
        done = [False] * n

        def push(i):
            downstream = out_of[pipes[i][1]]
            for j in downstream:
                area[j] += area[i] / len(downstream)
                path[j] = max(path[j], path[i] + pipes[j][2])
                indeg[j] -= 1
                if indeg[j] == 0:
                    queue.append(j)

        while queue:
            i = queue.popleft()
            done[i] = True
            push(i)
        # Pipes in loops never reach indeg 0; keep what has accumulated so far.
        return area, path

    def simulate(self, intensity_mm_hr: float, duration_min: float) -> List[Dict]:
        i = max(float(intensity_mm_hr), 0.0)
        dur = max(float(duration_min), 0.0)
        c = C_MIN + (C_MAX - C_MIN) * min(dur / C_SATURATION_MIN, 1.0)
        results = []
        for k in range(len(self.features)):
            cap = self.capacity[k]
            tc_min = INLET_TIME_MIN + self.flow_path_m[k] / FLOW_VELOCITY_MS / 60.0
            frac = min(dur / tc_min, 1.0) if tc_min > 0 else 1.0
            q_in = c * i * self.upstream_area_m2[k] * frac / 3.6e6
            load = (q_in / cap) if cap else None
            overflow_m3 = max(q_in - cap, 0.0) * dur * 60.0 if cap else None
            results.append({
                "inflow_m3s": q_in,
                "capacity_m3s": cap,
                "load_ratio": load,
                "overflow_m3": overflow_m3,
                "alert": alert_for(load),
            })
        return results


@lru_cache(maxsize=1)
def get_drainage_network() -> DrainageNetwork:
    return DrainageNetwork()


@lru_cache(maxsize=64)
def simulate_cached(intensity_mm_hr: float, duration_min: float):
    return get_drainage_network().simulate(intensity_mm_hr, duration_min)


def overall_alert(counts: Dict[str, int]) -> str:
    """Worst level reached by at least 20% of the assessed pipes."""
    total = sum(v for k, v in counts.items() if k != "UNKNOWN")
    if not total:
        return "UNKNOWN"
    cum = 0
    for level in ("RED", "ORANGE", "YELLOW"):
        cum += counts.get(level, 0)
        if cum / total >= 0.20:
            return level
    return "GREEN"


def run_simulation(intensity_mm_hr: float, duration_min: float,
                   lat: Optional[float] = None, lon: Optional[float] = None,
                   radius_m: Optional[float] = None) -> Dict:
    net = get_drainage_network()
    sim = simulate_cached(round(float(intensity_mm_hr), 1), round(float(duration_min), 1))

    def in_area(f):
        if lat is None or lon is None or not radius_m:
            return True
        mx, my = _metres_per_deg(lat)
        dx, dy = (f["mid"][0] - lon) * mx, (f["mid"][1] - lat) * my
        return dx * dx + dy * dy <= radius_m * radius_m

    features, counts, overflow_total = [], defaultdict(int), 0.0
    for f, r in zip(net.features, sim):
        local = in_area(f)
        if local:
            counts[r["alert"]] += 1
            overflow_total += r["overflow_m3"] or 0.0
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": f["coords"]},
            "properties": {
                "segment_id": f["segment_id"],
                "ward": f["ward"],
                "pipe_diameter_mm": f["pipe_diameter_mm"],
                "alert": r["alert"],
                "load_pct": round(r["load_ratio"] * 100) if r["load_ratio"] is not None else None,
                "inflow_m3s": round(r["inflow_m3s"], 4),
                "capacity_m3s": round(r["capacity_m3s"], 4) if r["capacity_m3s"] else None,
                "overflow_m3": round(r["overflow_m3"]) if r["overflow_m3"] else 0,
                "in_area": local,
            },
        })

    level = overall_alert(counts)
    return {
        "scenario": {"rainfall_mm_hr": intensity_mm_hr, "duration_min": duration_min,
                     "total_rainfall_mm": round(intensity_mm_hr * duration_min / 60.0, 1)},
        "area": ({"lat": lat, "lon": lon, "radius_m": radius_m} if radius_m and lat is not None else None),
        "summary": {
            "overall_alert": level,
            "overall_meaning": ALERT_MEANING.get(level, "No pipes with known capacity in this area"),
            "pipes_assessed": sum(counts.values()),
            "counts": {k: counts.get(k, 0) for k in ("RED", "ORANGE", "YELLOW", "GREEN", "UNKNOWN")},
            "estimated_overflow_m3": round(overflow_total),
        },
        "alert_meaning": ALERT_MEANING,
        "assumptions": ASSUMPTIONS,
        "network": {"type": "FeatureCollection", "features": features},
    }


def load_ratio_for_segment(segment_id: Optional[str], intensity_mm_hr: float, duration_min: float) -> Optional[float]:
    if segment_id is None:
        return None
    net = get_drainage_network()
    k = net.index_by_segment.get(segment_id)
    if k is None:
        return None
    return simulate_cached(round(float(intensity_mm_hr), 1), round(float(duration_min), 1))[k]["load_ratio"]
