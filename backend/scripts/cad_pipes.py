"""
Page-space pipe reconstruction for KMC AutoCAD ward sheets.

AutoCAD exports a dashed pipe as hundreds of separate short strokes. This module
  1. chains dashes back into continuous polylines (endpoint gap + collinearity,
     mutual-best matching, so branches at T-junctions stay separate),
  2. reassembles diameter labels that were exported glyph-by-glyph
     ("(", "3", "0", "0", "m", "m", "Ø", ")") into strings,
  3. assigns each chain a diameter from labels lying alongside and parallel to it.
     A chain with conflicting labels is split between them. A chain with no label
     keeps diameter None and is never guessed.

Everything here is in PDF page coordinates. Georeferencing happens separately
(ground control points picked in QGIS).
"""

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import substring
from shapely.strtree import STRtree

XY = Tuple[float, float]


# ------------------------------------------------------------------ chaining

@dataclass
class Chain:
    pts: List[XY]
    layer: str
    style: tuple
    n_dashes: int
    labels: List[Tuple[int, float]] = field(default_factory=list)  # (diameter, position along)
    diameter_mm: Optional[int] = None
    diameter_source: Optional[str] = None
    label_conflict: bool = False
    kind: Optional[str] = None  # caller-defined category, preserved through splits


def _dir(a: XY, b: XY) -> Tuple[float, float]:
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = math.hypot(dx, dy) or 1.0
    return dx / n, dy / n


def chain_dashes(dashes: Sequence[Tuple[List[XY], str, tuple]], max_gap: float,
                 max_angle_deg: float = 30.0) -> List[Chain]:
    """dashes: (points, layer, style). Joins end-to-end continuations only."""
    items = [(pts, lay, sty) for pts, lay, sty in dashes if len(pts) >= 2 and pts[0] != pts[-1]]
    n = len(items)
    # endpoint k: dash i = k // 2, end = k % 2 (0 = start, 1 = end)
    ep = np.array([items[k // 2][0][0] if k % 2 == 0 else items[k // 2][0][-1] for k in range(2 * n)])
    outward = []
    for k in range(2 * n):
        pts = items[k // 2][0]
        outward.append(_dir(pts[1], pts[0]) if k % 2 == 0 else _dir(pts[-2], pts[-1]))
    outward = np.array(outward)
    cos_max = math.cos(math.radians(max_angle_deg))
    tree = STRtree([Point(p) for p in ep])

    best: Dict[int, Tuple[float, int]] = {}
    for k in range(2 * n):
        cand = tree.query(Point(ep[k]).buffer(max_gap))
        for j in cand:
            j = int(j)
            if j // 2 == k // 2:
                continue
            # same layer & style only
            if items[j // 2][1] != items[k // 2][1] or items[j // 2][2] != items[k // 2][2]:
                continue
            gap = ep[j] - ep[k]
            d = float(np.hypot(*gap))
            if d > max_gap:
                continue
            # continuation: other dash's outward direction opposes ours
            if float(np.dot(outward[k], -outward[j])) < cos_max:
                continue
            # the gap itself must run along the direction (skip for tiny gaps)
            if d > 0.3 and float(np.dot(outward[k], gap / d)) < cos_max:
                continue
            ang = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(outward[k], -outward[j]))))))
            score = d + 0.05 * ang
            if k not in best or score < best[k][0]:
                best[k] = (score, j)
    link = {}
    for k, (_, j) in best.items():
        if j in best and best[j][1] == k:
            link[k] = j

    seen = set()
    chains = []
    for start in range(n):
        if start in seen:
            continue
        # walk to one end of this chain
        i, end_in = start, 0
        visited_walk = {start}
        while True:
            k = 2 * i + end_in
            if k in link and link[k] // 2 not in visited_walk:
                j = link[k]
                i, end_in = j // 2, 1 - (j % 2)
                visited_walk.add(i)
            else:
                break
        # now i is at an end; its free end is (2*i + end_in); traverse from there
        pts: List[XY] = []
        cur, entry = i, end_in  # entry: the free end index (0/1)
        count = 0
        while True:
            seen.add(cur)
            count += 1
            dpts = items[cur][0] if entry == 0 else list(reversed(items[cur][0]))
            pts.extend(dpts)
            exit_k = 2 * cur + (1 - entry)
            if exit_k in link and link[exit_k] // 2 not in seen:
                j = link[exit_k]
                cur, entry = j // 2, j % 2
            else:
                break
        chains.append(Chain(pts=pts, layer=items[i][1], style=items[i][2], n_dashes=count))
    return chains


# -------------------------------------------------------------------- labels

@dataclass
class Label:
    text: str
    x: float          # centre
    y: float
    angle_deg: float
    diameter_mm: Optional[int]
    unit: str = "mm"


def assemble_labels(glyphs, decode=None, max_step: float = 8.0, max_angle_diff: float = 12.0) -> List[Label]:
    """Group per-glyph text items into strings along their baseline direction.

    glyphs: TextItem-like objects with .text .x .y .angle_deg .font_size.
    Whole-string items (already containing 'mm') pass through unchanged.
    """
    out: List[Label] = []
    singles = []
    for g in glyphs:
        t = decode(g.text) if decode else g.text
        if len(t.strip()) > 1:
            out.append(_label(t, [(g.x, g.y)], g.angle_deg, g.font_size))
        else:
            singles.append((t, g))
    used = [False] * len(singles)
    pts = [Point(g.x, g.y) for _, g in singles]
    tree = STRtree(pts) if pts else None
    for s_i, (t, g) in enumerate(singles):
        if used[s_i] or t != "(":
            continue
        a = math.radians(g.angle_deg)
        ux, uy = math.cos(a), math.sin(a)
        group = [(0.0, t, g)]
        used[s_i] = True
        cur = (g.x, g.y)
        while True:
            nxt = None
            for j in tree.query(Point(cur).buffer(max_step)):
                j = int(j)
                if used[j]:
                    continue
                tj, gj = singles[j]
                dx, dy = gj.x - g.x, gj.y - g.y
                along = dx * ux + dy * uy
                perp = abs(-dx * uy + dy * ux)
                if along <= group[-1][0] or perp > 0.5 * g.font_size:
                    continue
                if abs(((gj.angle_deg - g.angle_deg + 180) % 360) - 180) > max_angle_diff:
                    continue
                if nxt is None or along < nxt[0]:
                    nxt = (along, j)
            if nxt is None:
                break
            used[nxt[1]] = True
            tj, gj = singles[nxt[1]]
            group.append((nxt[0], tj, gj))
            cur = (gj.x, gj.y)
            if tj == ")":
                break
        text = "".join(t for _, t, _ in group)
        out.append(_label(text, [(gg.x, gg.y) for _, _, gg in group], g.angle_deg, g.font_size))
    return out


DIA_MM = re.compile(r"(\d{3,4})\s*mm", re.I)
DIA_IN = re.compile(r"(\d{1,2})\s*[\"”]\s*[Øø]")


def _label(text, anchors, angle, fs) -> Label:
    xs = [p[0] for p in anchors]
    ys = [p[1] for p in anchors]
    a = math.radians(angle)
    # centre: midpoint of glyph anchors, lifted by ~0.35 em toward the glyph middle
    cx = (min(xs) + max(xs)) / 2 - 0.35 * fs * math.sin(a)
    cy = (min(ys) + max(ys)) / 2 + 0.35 * fs * math.cos(a)
    if len(anchors) == 1:  # whole string: anchor is baseline start
        half = 0.27 * fs * len(text)
        cx = anchors[0][0] + half * math.cos(a) - 0.35 * fs * math.sin(a)
        cy = anchors[0][1] + half * math.sin(a) + 0.35 * fs * math.cos(a)
    m = DIA_MM.search(text)
    dia, unit = (int(m.group(1)), "mm") if m else (None, "mm")
    if dia is None:
        mi = DIA_IN.search(text)
        if mi:
            dia, unit = int(mi.group(1)), "in"
    return Label(text=text, x=cx, y=cy, angle_deg=angle, diameter_mm=dia, unit=unit)


# ---------------------------------------------------------------- assignment

def assign_diameters(chains: List[Chain], labels: List[Label], max_dist: float,
                     max_angle_deg: float = 25.0, ambiguity_ratio: float = 1.5) -> Dict:
    """Attach each diameter label to the nearest parallel chain.

    Labels in these sheets sit beside the road edge, 5-30 pt from the pipe.
    To avoid giving a label to the pipe in the next street, a label is dropped
    as ambiguous when a second parallel chain carrying a different candidate is
    within `ambiguity_ratio` x the best distance.
    """
    lines = [LineString(c.pts) for c in chains]
    tree = STRtree(lines)
    stats = defaultdict(int)
    for lab in labels:
        if lab.diameter_mm is None or lab.unit != "mm":
            stats["labels_unparsed_or_inch"] += 1
            continue
        p = Point(lab.x, lab.y)
        la = math.radians(lab.angle_deg)
        cands = []
        for j in tree.query(p.buffer(max_dist)):
            j = int(j)
            ln = lines[j]
            d = ln.distance(p)
            if d > max_dist:
                continue
            s = ln.project(p)
            a0 = ln.interpolate(max(0.0, s - 3))
            a1 = ln.interpolate(min(ln.length, s + 3))
            seg_ang = math.atan2(a1.y - a0.y, a1.x - a0.x)
            diff = abs(math.degrees((seg_ang - la + math.pi / 2) % math.pi - math.pi / 2))
            if diff > max_angle_deg:
                continue
            cands.append((d, j, s))
        if not cands:
            stats["labels_unmatched"] += 1
            continue
        cands.sort()
        d0, j0, s0 = cands[0]
        rivals = [c for c in cands[1:] if c[0] < ambiguity_ratio * max(d0, 2.0)
                  and lines[c[1]].distance(lines[j0]) > 2.0]  # a genuinely different pipe
        if rivals:
            stats["labels_ambiguous"] += 1
            continue
        chains[j0].labels.append((lab.diameter_mm, s0))
        stats["labels_matched"] += 1
    return dict(stats)


def resolve_chains(chains: List[Chain]) -> List[Chain]:
    """One diameter per output chain; split chains that carry different labels."""
    out: List[Chain] = []
    for c in chains:
        dias = sorted(set(d for d, _ in c.labels))
        if len(dias) <= 1:
            c.diameter_mm = dias[0] if dias else None
            c.diameter_source = "map_label" if dias else None
            out.append(c)
            continue
        # Split between consecutive labels that differ (at the midpoint).
        ln = LineString(c.pts)
        labs = sorted(c.labels, key=lambda t: t[1])
        cuts = [0.0]
        segs_dia = [labs[0][0]]
        for (d0, s0), (d1, s1) in zip(labs[:-1], labs[1:]):
            if d1 != d0:
                cuts.append((s0 + s1) / 2)
                segs_dia.append(d1)
        cuts.append(ln.length)
        for k, dia in enumerate(segs_dia):
            part = substring(ln, cuts[k], cuts[k + 1])
            if part.length <= 0 or part.geom_type != "LineString":
                continue
            out.append(Chain(pts=list(part.coords), layer=c.layer, style=c.style,
                             n_dashes=0, labels=[(dia, 0.0)], diameter_mm=dia,
                             diameter_source="map_label_split", label_conflict=True, kind=c.kind))
    return out
