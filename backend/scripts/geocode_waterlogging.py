"""
Build data/historical/waterlogging_points.csv from the parsed KMC 2017 pocket list.

    ../.venv/bin/python scripts/geocode_waterlogging.py --provider none
        Writes every record with EMPTY coordinates. No network access. The
        loader counts these as "unlocated" and never uses them in lookups.

    ../.venv/bin/python scripts/geocode_waterlogging.py --provider nominatim --i-accept-nominatim-policy
        One-off geocode with the public OSM Nominatim server, following its policy
        (https://operations.osmfoundation.org/policies/nominatim/): a single
        thread, at most 1 request/s (1.1 s spacing), an identifying User-Agent,
        cached results so no query repeats, and attribution
        "(c) OpenStreetMap contributors, ODbL". Run it once; don't schedule it.

    ../.venv/bin/python scripts/geocode_waterlogging.py --provider mapbox
        Mapbox Geocoding v6 with permanent=true. Storing results is only
        allowed with permanent geocoding, which is billed separately. Needs
        MAPBOX_ACCESS_TOKEN.

A candidate location is accepted only if all of these hold:
  - the query has a distinctive name token (not just "Village Road" or "East Park")
  - the candidate's own name contains every distinctive token (fuzzy match
    allows for transliteration)
  - it lies inside the KMC boundary (data/gis/kmc_boundary.geojson)
  - it isn't ambiguous: no two good candidates more than 1 km apart
  - it isn't too vague: not a city-level feature, and its extent is no more
    than MAX_UNCERTAINTY_M. Streets have no extent; they are kept as
    "<provider>_street_centroid" so they're never read as a precise spot.

LICENSING: Mapbox permanent results are for your own use only, with no
distribution. waterlogging_points.csv, geocode_cache.json and
geocode_review.csv are gitignored for that reason, because the repo is public.
Anything else keeps empty coordinates and a geocode_method saying why. Nothing
is guessed. Rows listed in data/historical/manual_coordinates.csv (columns:
borough, source_sn, latitude, longitude, notes) override the geocoder with
coordinates you verified yourself.
"""

import argparse
import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
from pyproj import Geod, Transformer
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, settings  # noqa: E402

RAW = DATA_DIR / "historical" / "kmc_2017_pockets_raw.csv"
OUT = DATA_DIR / "historical" / "waterlogging_points.csv"
MANUAL = DATA_DIR / "historical" / "manual_coordinates.csv"
CACHE = DATA_DIR / "historical" / "geocode_cache.json"
REVIEW = DATA_DIR / "historical" / "geocode_review.csv"

YEAR = 2017
SEVERITY = "listed_major_pocket"  # the section title; the PDF gives no per-row severity
MAX_UNCERTAINTY_M = 750.0
AMBIGUITY_M = 1000.0
USER_AGENT = "FloodWatch/0.1 (one-off geocode of KMC 2017 waterlogging pocket list)"
KOLKATA_VIEWBOX = (88.24, 22.64, 88.46, 22.43)  # lon1, lat1, lon2, lat2

GEOD = Geod(ellps="WGS84")
TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32645", always_xy=True)

COLUMNS = ["location_name", "ward", "latitude", "longitude", "source", "year", "severity", "notes",
           "geocode_method", "geocode_query", "geocode_display_name", "location_uncertainty_m",
           "borough", "source_sn", "source_page"]


# ------------------------------------------------------------- query text

def clean_query(location_text: str) -> str:
    """Primary place name for geocoding: drop parentheticals, premises numbers, 'portion'."""
    t = re.sub(r"\([^)]*\)?", " ", location_text)          # (portion), (from ... to ...), (near ...)
    t = re.split(r"\s*&\s*|\s+and\s+(?=[A-Z])", t)[0]       # first of "A & B"
    # "X near Y" / "X beside Y" / "X opp. Y": keep X, don't merge two places into one query
    t = re.split(r"(?i)\s+(?:near|beside|opp\.?|opposite|infront of|in front of|towards)\s+", t)[0]
    t = re.sub(r"(?i)\b(portion|part|jn\.?|junction of|crossing of)\b", " ", t)
    # Leading premises numbers are stripped only when comma-separated ("46, Middle Road",
    # "2A, 2B, 2C Chatu Babu Lane", "4,5 Netaji Nagar"). Numbers that are part of a
    # place name ("22 Bigha", "64 Pally", "8 No. Sahid Nagar") are kept.
    t = re.sub(r"^\s*(?:\d[0-9A-Za-z/\-]*\s*,\s*)+(?:\d[0-9A-Za-z/\-]*\s+)?", "", t)
    t = re.sub(r"\bRd\b\.?", "Road", t)
    t = re.sub(r"\bSt\b\.?", "Street", t)
    t = re.sub(r"\bLn\b\.?", "Lane", t)
    return re.sub(r"\s+", " ", t).strip(" ,.-")


# ------------------------------------------------------------- providers

class Nominatim:
    name = "nominatim"
    url = "https://nominatim.openstreetmap.org/search"
    min_interval = 1.1

    def __init__(self):
        self._last = 0.0

    def search(self, q: str) -> List[Dict]:
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        params = {"q": f"{q}, Kolkata, West Bengal, India", "format": "jsonv2", "limit": 5,
                  "countrycodes": "in", "viewbox": ",".join(map(str, KOLKATA_VIEWBOX)), "bounded": 1}
        req = urllib.request.Request(f"{self.url}?{urllib.parse.urlencode(params)}",
                                     headers={"User-Agent": USER_AGENT})
        try:
            data = json.load(urllib.request.urlopen(req, timeout=30))
        finally:
            self._last = time.monotonic()
        out = []
        for d in data:
            s, n, w, e = (float(v) for v in d["boundingbox"])
            out.append({"lat": float(d["lat"]), "lon": float(d["lon"]), "bbox": (w, s, e, n),
                        "display_name": d.get("display_name"), "kind": f"{d.get('category')}:{d.get('type')}"})
        return out


class FatalProviderError(RuntimeError):
    """Auth/quota errors: abort the whole run instead of caching bad results."""


class Mapbox:
    name = "mapbox"
    url = "https://api.mapbox.com/search/geocode/v6/forward"
    min_interval = 0.1  # well under Mapbox's per-minute rate limit

    def __init__(self, token: str):
        self.token = token
        self._last = 0.0

    def search(self, q: str) -> List[Dict]:
        import urllib.error
        w, n, e, s = KOLKATA_VIEWBOX
        params = {"q": f"{q}, Kolkata", "country": "in", "limit": 5, "permanent": "true",
                  "bbox": f"{w},{s},{e},{n}", "access_token": self.token}
        for attempt in range(4):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                data = json.load(urllib.request.urlopen(f"{self.url}?{urllib.parse.urlencode(params)}", timeout=30))
                break
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403):
                    raise FatalProviderError(f"Mapbox rejected the token (HTTP {exc.code})") from None
                if exc.code == 429 and attempt < 3:
                    time.sleep(5 * (attempt + 1))
                    continue
                raise
            finally:
                self._last = time.monotonic()
        out = []
        for f in data.get("features", []):
            lon, lat = f["geometry"]["coordinates"]
            p = f.get("properties", {})
            bb = p.get("bbox") or f.get("bbox")
            out.append({"lat": lat, "lon": lon, "bbox": tuple(bb) if bb else None,
                        "name": p.get("name") or "", "display_name": p.get("full_address") or p.get("name"),
                        "kind": p.get("feature_type")})
        return out


# ------------------------------------------------------------- selection

def extent_m(bbox) -> Optional[float]:
    """Half-diagonal of a (w, s, e, n) bbox in metres, a location-uncertainty proxy."""
    if not bbox:
        return None
    w, s, e, n = bbox
    _, _, d = GEOD.inv(w, s, e, n)
    return d / 2.0


# Words that don't identify a place on their own. A query needs at least one
# other ("distinctive") token, and every distinctive token must appear in the
# candidate's own name. This stops "Hossenpur 2nd Lane" from matching some
# random "2nd Lane" elsewhere in the city.
GENERIC = {
    "road", "rd", "lane", "ln", "street", "st", "sarani", "sarak", "avenue", "ave", "row", "place",
    "bye", "main", "no", "the", "of", "and", "near", "bustee", "colony", "park", "block", "bl",
    "area", "gali", "kolkata", "calcutta", "east", "west", "north", "south", "new", "old", "village",
    "portion", "part", "crossing", "more", "station", "garden", "gardens", "nagar", "pally", "palli",
    "para", "bagan", "abasan", "housing", "project", "scheme", "phase", "market", "bazar", "bazaar",
    "township", "terrace", "path",
}
ORDINAL = re.compile(r"^\d+(st|nd|rd|th)?$")
CITY_LEVEL_TYPES = {"country", "region", "postcode", "district", "place"}
# Street-type words: "Ripon Lane" and "Ripon Street" are different streets.
STREET_TYPE = {"road": "road", "rd": "road", "lane": "lane", "ln": "lane", "street": "street", "st": "street",
               "place": "place", "row": "row", "avenue": "avenue", "ave": "avenue", "sarani": "sarani",
               "sarak": "sarani", "terrace": "terrace"}

# Post-pass consistency: a listed ward is ~1-3 km across and a borough ~3-8 km.
# A point far from the other points of its own ward/borough is almost certainly
# a same-named place elsewhere, so it goes back to review.
WARD_OUTLIER_M = 2500.0
BOROUGH_OUTLIER_M = 6000.0


def _tokens(s: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", s.lower())]


def distinctive(q: str) -> List[str]:
    return [t for t in _tokens(q) if t not in GENERIC and not ORDINAL.match(t) and len(t) > 2]


def name_matches(q: str, cand_name: str) -> bool:
    from difflib import SequenceMatcher
    want = distinctive(q)
    have = _tokens(cand_name)
    if not want:
        return False
    # Transliteration varies (Hossenpur/Hossainpur, Sreehari/Srihari), so match fuzzily per token.
    # Also try joined forms: "Martin Para" vs "Martinpara".
    joined = "".join(have)
    if not all(any(SequenceMatcher(None, w, h).ratio() >= 0.8 for h in have) or w in joined for w in want):
        return False
    q_types = {STREET_TYPE[t] for t in _tokens(q) if t in STREET_TYPE}
    c_types = {STREET_TYPE[t] for t in have if t in STREET_TYPE}
    # If both name a street type, they must share one ("Lane" vs "Street" = different street).
    return not (q_types and c_types and not (q_types & c_types))


def _clusters(pts: List[Tuple[float, float]], limit: float) -> List[List[int]]:
    """Single-linkage clusters of (lat, lon) points: two points link if within `limit` metres."""
    n = len(pts)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a in range(n):
        for b in range(a + 1, n):
            _, _, d = GEOD.inv(pts[a][1], pts[a][0], pts[b][1], pts[b][0])
            if d <= limit:
                parent[find(a)] = find(b)
    groups: Dict[int, List[int]] = {}
    for a in range(n):
        groups.setdefault(find(a), []).append(a)
    return sorted(groups.values(), key=len, reverse=True)


def consistency_outliers(rows: List[Dict]) -> Dict[int, str]:
    """Row index -> reason, for located rows that disagree with their own ward/borough.

    The records of one ward (or borough) should sit together. Within each group
    the located points are clustered. Only a strict majority cluster is trusted:
      - points outside the single largest cluster are outliers
      - on a tie (e.g. 2 points far apart, or 3 singletons), no side can be
        trusted, so all of them go to review
    A group with a single located point can't be checked and is left alone.
    """
    out: Dict[int, str] = {}
    located = [(i, r) for i, r in enumerate(rows) if r["latitude"] != ""]

    def _check(key_fn, limit, label):
        groups: Dict = {}
        for i, r in located:
            for k in key_fn(r):
                groups.setdefault(k, []).append(i)
        for k, idx in groups.items():
            if len(idx) < 2:
                continue
            pts = [(float(rows[i]["latitude"]), float(rows[i]["longitude"])) for i in idx]
            cl = _clusters(pts, limit)
            tie = len(cl) > 1 and len(cl[0]) == len(cl[1])
            keep = set() if tie else set(cl[0])
            for pos, i in enumerate(idx):
                if pos not in keep and i not in out:
                    out[i] = (f"{label} {k}: conflicts with the other located records of the same {label} "
                              f"({'no majority' if tie else 'outside the majority cluster'}, "
                              f"link distance {limit / 1000:.1f} km)")
    _check(lambda r: [w.strip() for w in str(r["ward"]).split(",")], WARD_OUTLIER_M, "ward")
    _check(lambda r: [r["borough"]], BOROUGH_OUTLIER_M, "borough")
    return out


def choose(cands: List[Dict], boundary, query: str = "") -> Dict:
    """Pick one candidate or explain why none is acceptable. Pure function (testable)."""
    if not cands:
        return {"status": "not_found"}
    if query and not distinctive(query):
        return {"status": "too_generic"}
    inside = [c for c in cands if boundary is None or boundary.covers(Point(*TO_UTM.transform(c["lon"], c["lat"])))]
    if not inside:
        return {"status": "outside_kmc"}
    named = [c for c in inside if not query or name_matches(query, c.get("name") or c.get("display_name") or "")]
    if not named:
        return {"status": "name_mismatch", "returned": [c.get("name") for c in inside][:3]}
    precise = [c for c in named if c.get("kind") not in CITY_LEVEL_TYPES
               and (extent_m(c.get("bbox")) or 0.0) <= MAX_UNCERTAINTY_M]
    if not precise:
        return {"status": "too_vague"}
    best = precise[0]
    for other in precise[1:]:
        _, _, d = GEOD.inv(best["lon"], best["lat"], other["lon"], other["lat"])
        if d > AMBIGUITY_M:
            return {"status": "ambiguous", "spread_m": round(d)}
    unc = extent_m(best.get("bbox"))
    return {"status": "ok", "lat": round(best["lat"], 6), "lon": round(best["lon"], 6),
            "uncertainty_m": round(unc) if unc is not None else None,
            # Streets come back as a single point with no extent: the centre of a
            # road of unknown length. Kept, but labelled so it's never read as a precise spot.
            "method_suffix": "street_centroid" if best.get("kind") == "street" and unc is None else "auto",
            # The matched feature name first; full_address can show an alias (e.g. "Ripon Ln"
            # comes back with full_address "Muzaffar Ahmed Street").
            "display_name": " | ".join(x for x in (best.get("name"), best.get("display_name")) if x),
            "kind": best.get("kind")}


# ------------------------------------------------------------- main

def load_boundary():
    p = settings.KMC_BOUNDARY_GEOJSON_PATH
    if not p.exists():
        raise SystemExit(f"KMC boundary missing ({p}); run scripts/fetch_kmc_boundary.py first")
    return gpd.read_file(p).to_crs("EPSG:32645").union_all()


def load_manual() -> Dict:
    if not MANUAL.exists():
        return {}
    out = {}
    for r in csv.DictReader(MANUAL.open(encoding="utf-8")):
        out[(int(r["borough"]), int(r["source_sn"]))] = r
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["none", "nominatim", "mapbox"], required=True)
    ap.add_argument("--i-accept-nominatim-policy", action="store_true")
    args = ap.parse_args(argv)
    if args.provider == "nominatim" and not args.i_accept_nominatim_policy:
        print("Refusing: read https://operations.osmfoundation.org/policies/nominatim/ and re-run with "
              "--i-accept-nominatim-policy")
        return 2
    provider = None
    if args.provider == "nominatim":
        provider = Nominatim()
    elif args.provider == "mapbox":
        tok = settings.MAPBOX_ACCESS_TOKEN
        if tok is None or not tok.get_secret_value().strip():
            print("Refusing: MAPBOX_ACCESS_TOKEN not configured")
            return 2
        provider = Mapbox(tok.get_secret_value())

    boundary = load_boundary() if provider else None
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    manual = load_manual()
    raw = list(csv.DictReader(RAW.open(encoding="utf-8")))
    rows, review = [], []
    for r in raw:
        key = (int(r["borough"]), int(r["sn"]))
        q = clean_query(r["location_text"])
        rec = {
            "location_name": r["location_text"], "ward": r["wards"], "latitude": "", "longitude": "",
            "source": r["source"], "year": YEAR, "severity": SEVERITY,
            "notes": f"Borough {r['borough']}, SN {r['sn']}, PDF page {r['page']}",
            "geocode_method": "not_geocoded", "geocode_query": q, "geocode_display_name": "",
            "location_uncertainty_m": "", "borough": r["borough"], "source_sn": r["sn"], "source_page": r["page"],
        }
        if key in manual:
            m = manual[key]
            rec.update(latitude=m["latitude"], longitude=m["longitude"], geocode_method="manual_verified",
                       notes=f"{rec['notes']}; manual: {m.get('notes', '')}".strip("; "))
        elif provider:
            ck = f"{provider.name}|{q}"
            cands = cache.get(ck)
            if ck not in cache and distinctive(q):  # don't pay for queries that can't be accepted anyway
                try:
                    cands = provider.search(q)
                    cache[ck] = cands
                    CACHE.write_text(json.dumps(cache, indent=1))  # persist after every call
                except FatalProviderError as exc:
                    print(f"ABORT: {exc}")
                    return 3
                except Exception as exc:  # transient: not cached, so a re-run retries it
                    print(f"  lookup failed for {q!r}: {exc}")
                    cands = None
            if not distinctive(q):
                res = {"status": "too_generic"}
            elif cands is None:
                res = {"status": "lookup_error"}
            else:
                res = choose(cands, boundary, q)
            if res["status"] == "ok":
                rec.update(latitude=res["lat"], longitude=res["lon"],
                           geocode_method=f"{provider.name}_{res['method_suffix']}",
                           geocode_display_name=res.get("display_name") or "",
                           location_uncertainty_m=res.get("uncertainty_m") or "")
            else:
                rec["geocode_method"] = f"{provider.name}_{res['status']}"
                review.append({**rec, "detail": json.dumps({k: v for k, v in res.items() if k != "status"})})
        rows.append(rec)

    if provider:
        for i, why in consistency_outliers(rows).items():
            r = rows[i]
            if r["geocode_method"] == "manual_verified":
                continue  # your own coordinates are never overridden
            detail = json.dumps({"reason": why, "rejected_lat": r["latitude"], "rejected_lon": r["longitude"],
                                 "rejected_method": r["geocode_method"]})
            r.update(latitude="", longitude="", geocode_method=f"{provider.name}_location_outlier",
                     location_uncertainty_m="")
            review.append({**r, "detail": detail})

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    located = sum(1 for r in rows if r["latitude"] != "")
    print(f"wrote {OUT}: {len(rows)} records, {located} located, {len(rows) - located} without coordinates")
    if review:
        with REVIEW.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS + ["detail"])
            w.writeheader()
            w.writerows(review)
        print(f"wrote {REVIEW}: {len(review)} records needing manual coordinates")
        from collections import Counter
        print("  reasons:", dict(Counter(r["geocode_method"] for r in review)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
