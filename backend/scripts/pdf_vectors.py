"""
Low-level vector/text reader for layered (OCG) PDF maps.

Walks a page's content stream with a minimal graphics-state machine
(q/Q, cm, RG/rg/G/g/K/k, w, m/l/c/v/y/re/h, paint ops) and yields:
  - PathItem: one painted path (a list of subpaths) in PAGE coordinates
    (the CTM is applied), with its optional-content layer, stroke/fill colour,
    line width and paint operator.
  - TextItem: one text-show operation with its decoded text, page position,
    layer and font size (decoding delegated to pypdf).

It deliberately does not rasterise anything. Curves (c/v/y) are reduced to
their end points: sewer lines in these maps are polylines, and curve control
points are not positions on the line.

Form XObjects are not descended into. The KMC maps inspected so far draw their
vectors directly in the page stream (the only XObjects are basemap images).
The walker counts any Form XObject it skips, so callers can fail loudly if one
turns up.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from pypdf import PdfReader
from pypdf.generic import ContentStream

Matrix = Tuple[float, float, float, float, float, float]
IDENTITY: Matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def mat_mul(m1: Matrix, m2: Matrix) -> Matrix:
    """PDF convention: m1 applied first, then m2."""
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    return (
        a1 * a2 + b1 * c2, a1 * b2 + b1 * d2,
        c1 * a2 + d1 * c2, c1 * b2 + d1 * d2,
        e1 * a2 + f1 * c2 + e2, e1 * b2 + f1 * d2 + f2,
    )


def apply(m: Matrix, x: float, y: float) -> Tuple[float, float]:
    a, b, c, d, e, f = m
    return (a * x + c * y + e, b * x + d * y + f)


def _color(op: bytes, args) -> Optional[Tuple[float, ...]]:
    vals = tuple(round(float(v), 4) for v in args)
    if op in (b"G", b"g"):
        return (vals[0],) * 3
    if op in (b"RG", b"rg"):
        return vals
    if op in (b"K", b"k"):  # CMYK -> RGB (naive, sufficient for matching)
        c, m, y, k = vals
        return tuple(round((1 - v) * (1 - k), 4) for v in (c, m, y))
    return None


@dataclass
class PathItem:
    layer: Optional[str]
    layers: Tuple[str, ...]
    stroke: Optional[Tuple[float, ...]]
    fill: Optional[Tuple[float, ...]]
    width: float           # line width in page units (scaled by CTM)
    paint: str             # S, s, f, f*, B, B*, b, b*, n
    subpaths: List[List[Tuple[float, float]]]
    closed: List[bool]
    order: int             # draw order index

    @property
    def stroked(self) -> bool:
        return self.paint in ("S", "s", "B", "B*", "b", "b*")

    @property
    def filled(self) -> bool:
        return self.paint in ("f", "F", "f*", "B", "B*", "b", "b*")


@dataclass
class TextItem:
    layer: Optional[str]
    layers: Tuple[str, ...]
    text: str
    x: float
    y: float
    font_size: float
    angle_deg: float = 0.0


@dataclass
class PageVectors:
    paths: List[PathItem] = field(default_factory=list)
    texts: List[TextItem] = field(default_factory=list)
    skipped_form_xobjects: int = 0
    page_size: Tuple[float, float] = (0.0, 0.0)


def _oc_names(page) -> Dict[str, str]:
    res = page.get("/Resources") or {}
    props = res.get("/Properties")
    if not props:
        return {}
    props = props.get_object()
    return {
        str(k): str(v.get_object().get("/Name", "?")).replace("\x00", "").strip()
        for k, v in props.items()
    }


def read_page(reader: PdfReader, page_index: int) -> PageVectors:
    import math

    page = reader.pages[page_index]
    oc = _oc_names(page)
    mb = page.mediabox
    out = PageVectors(page_size=(float(mb.width), float(mb.height)))

    xobjs = (page.get("/Resources") or {}).get("/XObject")
    xobjs = xobjs.get_object() if xobjs else {}

    # ---------------- paths ----------------
    ctm = IDENTITY
    stroke = (0.0, 0.0, 0.0)
    fill = (0.0, 0.0, 0.0)
    width = 1.0
    gstack: List[tuple] = []
    mc_stack: List[Optional[str]] = []
    subpaths: List[List[Tuple[float, float]]] = []
    closed: List[bool] = []
    cur: List[Tuple[float, float]] = []
    order = 0

    def flush_cur():
        nonlocal cur
        if cur:
            subpaths.append(cur)
            closed.append(False)
        cur = []

    def layers_now():
        return tuple(s for s in mc_stack if s)

    for args, op in ContentStream(page.get_contents(), reader).operations:
        if op == b"q":
            gstack.append((ctm, stroke, fill, width))
        elif op == b"Q":
            if gstack:
                ctm, stroke, fill, width = gstack.pop()
        elif op == b"cm":
            ctm = mat_mul(tuple(float(v) for v in args), ctm)
        elif op in (b"RG", b"G", b"K"):
            stroke = _color(op, args)
        elif op in (b"rg", b"g", b"k"):
            fill = _color(op, args)
        elif op == b"w":
            width = float(args[0])
        elif op == b"BDC":
            name = None
            if len(args) > 1 and str(args[0]) == "/OC":
                name = oc.get(str(args[1]), str(args[1]))
            mc_stack.append(name)
        elif op == b"BMC":
            mc_stack.append(None)
        elif op == b"EMC":
            if mc_stack:
                mc_stack.pop()
        elif op == b"m":
            flush_cur()
            cur = [apply(ctm, float(args[0]), float(args[1]))]
        elif op == b"l":
            cur.append(apply(ctm, float(args[0]), float(args[1])))
        elif op in (b"c", b"v", b"y"):
            cur.append(apply(ctm, float(args[-2]), float(args[-1])))
        elif op == b"re":
            flush_cur()
            x, y, w_, h_ = (float(v) for v in args)
            pts = [(x, y), (x + w_, y), (x + w_, y + h_), (x, y + h_), (x, y)]
            subpaths.append([apply(ctm, *p) for p in pts])
            closed.append(True)
        elif op == b"h":
            if cur:
                cur.append(cur[0])
                subpaths.append(cur)
                closed.append(True)
                cur = []
        elif op in (b"S", b"s", b"f", b"F", b"f*", b"B", b"B*", b"b", b"b*", b"n"):
            flush_cur()
            if op != b"n" and subpaths:
                # Scale line width by the CTM (geometric mean of axis scales).
                sx = math.hypot(ctm[0], ctm[1])
                sy = math.hypot(ctm[2], ctm[3])
                lays = layers_now()
                out.paths.append(PathItem(
                    layer=lays[-1] if lays else None,
                    layers=lays,
                    stroke=stroke, fill=fill,
                    width=width * math.sqrt(sx * sy),
                    paint=op.decode(),
                    subpaths=subpaths, closed=closed, order=order,
                ))
                order += 1
            subpaths, closed = [], []
        elif op == b"Do":
            xo = xobjs.get(str(args[0]))
            if xo is not None and xo.get_object().get("/Subtype") == "/Form":
                out.skipped_form_xobjects += 1

    # ---------------- text (decoding via pypdf) ----------------
    tstack: List[Optional[str]] = []

    def before(op, args, cm, tm):
        if op == b"BDC":
            name = None
            if args and len(args) > 1 and str(args[0]) == "/OC":
                name = oc.get(str(args[1]), str(args[1]))
            tstack.append(name)
        elif op == b"BMC":
            tstack.append(None)
        elif op == b"EMC" and tstack:
            tstack.pop()

    def on_text(text, cm, tm, font_dict, font_size):
        if not text or not text.strip():
            return
        m = mat_mul(tuple(tm), tuple(cm))
        x, y = m[4], m[5]
        scale = math.hypot(m[0], m[1])
        lays = tuple(s for s in tstack if s)
        out.texts.append(TextItem(
            layer=lays[-1] if lays else None, layers=lays,
            text=text.strip(), x=x, y=y,
            font_size=(font_size or 1.0) * scale,
            angle_deg=math.degrees(math.atan2(m[1], m[0])),
        ))

    page.extract_text(visitor_operand_before=before, visitor_text=on_text)
    return out
