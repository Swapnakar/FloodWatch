"""
QA overlay: draw extracted page-space pipes over the sheet's reference render.

Colour = assigned diameter, black = no diameter (never guessed). The PNG goes
to data/processed/kmc_extracted/cad/qa/, for checking chaining and label
assignment before georeferencing.

Usage (from backend/): ../.venv/bin/python scripts/qa_cad_overlay.py [ward ...]
"""

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_kmc_cad import CAD_DIR, RENDER_DIR, display_size, page_to_display  # noqa: E402

COLOURS = {150: (0, 200, 200), 225: (0, 92, 230), 250: (0, 150, 255), 300: (255, 140, 0),
           375: (170, 0, 255), 400: (150, 90, 0), 450: (0, 170, 0), 500: (60, 60, 140),
           600: (255, 0, 200), 700: (130, 0, 60), 800: (120, 120, 0), 1000: (90, 90, 0), 1200: (0, 110, 90)}


def main():
    wards = [int(w) for w in sys.argv[1:]] or [36, 61, 104]
    out = CAD_DIR / "qa"
    out.mkdir(parents=True, exist_ok=True)
    Image.MAX_IMAGE_PIXELS = None
    for w in wards:
        page = json.loads((CAD_DIR / f"ward{w:03d}_page.json").read_text())
        rep, segs = page["report"], page["segments"]
        W, H = rep["page_size_pt"]
        rot = rep["page_rotate"]
        k = rep["render"]["px_per_pt"]
        DW, DH = display_size(W, H, rot)
        base = Image.open(RENDER_DIR / f"ward{w:03d}.png").convert("RGB")
        base = Image.blend(base, Image.new("RGB", base.size, "white"), 0.55)
        d = ImageDraw.Draw(base)
        for sg in segs:
            pts = []
            for x, y in sg["page_pts"]:
                u, v = page_to_display(x, y, W, H, rot)
                pts.append((u * k, (DH - v) * k))
            dia = sg["pipe_diameter_mm"]
            col = COLOURS.get(dia, (255, 0, 0)) if dia else (0, 0, 0)
            d.line(pts, fill=col, width=4 if sg["conduit_type"] == "pipe" else 2)
        base.save(out / f"ward{w:03d}_qa.png")
        print(out / f"ward{w:03d}_qa.png", base.size)


if __name__ == "__main__":
    main()
