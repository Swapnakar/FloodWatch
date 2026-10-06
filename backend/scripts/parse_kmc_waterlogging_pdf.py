"""
Parse Section C "Major Water Logging Pockets" of the KMC 2017 action plan PDF.

Source: data/historical/water_logging_09_06_2017.pdf
        ("Action Plan to Mitigate Flood, Cyclone & Water Logging 2017",
         The Kolkata Municipal Corporation, Section C, pages 17-28)

Output: data/historical/kmc_2017_pockets_raw.csv with one row per listed
        pocket: borough, sn, wards, location_text, page. There are NO
        coordinates here; see geocode_waterlogging.py.

The table is extracted as text lines. A row starts with "SN[.] WARD[,WARD] text".
Some rows wrap onto following lines, and some have SN, ward and text on
separate lines. A candidate row is only accepted if its SN is exactly the
previous SN + 1 within the borough, which prevents a wrapped line that happens
to start with a number from being read as a new row. The script checks that
every borough's rows are numbered 1..N with no gaps, and fails otherwise.
"""

import csv
import re
import sys
from pathlib import Path

from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR  # noqa: E402

PDF = DATA_DIR / "historical" / "water_logging_09_06_2017.pdf"
OUT = DATA_DIR / "historical" / "kmc_2017_pockets_raw.csv"
FIRST_PAGE, LAST_PAGE = 17, 28  # 1-based, inclusive (Section C)
SOURCE = ("KMC 'Action Plan to Mitigate Flood, Cyclone & Water Logging 2017', "
          "Section C Major Water Logging Pockets")

ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9,
         "X": 10, "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15, "XVI": 16}
BOROUGH_RE = re.compile(r"^\s*Borough\s*[–\-]\s*([IVX]+)\s*$", re.I)
HEADER_RE = re.compile(r"^\s*(SN\s+Wd\.?\s*No|C\.\s*Major Water Logging|MAJOR WATER LOGGING|\d{1,3}\s*$|2017\s*$)", re.I)
WARDS = r"(\d{1,3}(?:\s*,\s*\d{1,3})*)"
ROW_RE = re.compile(rf"^\s*(\d{{1,3}})\.?\s+{WARDS}\s+(\S.*?)\s*$")
SN_ONLY_RE = re.compile(r"^\s*(\d{1,3})\.?\s*$")
SN_WARD_RE = re.compile(rf"^\s*(\d{{1,3}})\.?\s+{WARDS}\s*$")
WARD_ONLY_RE = re.compile(rf"^\s*{WARDS}\s*$")


def _wards(s):
    ws = [int(w) for w in re.split(r"\s*,\s*", s.strip())]
    return ws if all(1 <= w <= 144 for w in ws) else None


def parse(pdf=PDF):
    reader = PdfReader(str(pdf))
    rows = []
    borough = None
    pending = None  # {"sn":..,"wards":..} waiting for text
    for pno in range(FIRST_PAGE, LAST_PAGE + 1):
        lines = (reader.pages[pno - 1].extract_text() or "").splitlines()
        for i, raw in enumerate(lines):
            line = raw.replace("\u00a0", " ").strip()
            if not line:
                continue
            m = BOROUGH_RE.match(line)
            if m:
                new = ROMAN[m.group(1).upper()]
                if new != borough:
                    borough, pending = new, None
                continue
            # page numbers etc. (a lone number is only a header when not expected as SN/ward)
            if HEADER_RE.match(line) and not (pending or SN_ONLY_RE.match(line) and _expects_sn(rows, borough, line)):
                continue
            if borough is None:
                continue
            next_sn = _next_sn(rows, borough)
            m = ROW_RE.match(line)
            if m and int(m.group(1)) == next_sn and _wards(m.group(2)):
                rows.append({"borough": borough, "sn": next_sn, "wards": _wards(m.group(2)),
                             "location_text": m.group(3), "page": pno})
                pending = None
                continue
            m = SN_WARD_RE.match(line)
            if m and int(m.group(1)) == next_sn and _wards(m.group(2)):
                pending = {"sn": next_sn, "wards": _wards(m.group(2)), "page": pno}
                continue
            m = SN_ONLY_RE.match(line)
            if m and int(m.group(1)) == next_sn and pending is None:
                pending = {"sn": next_sn, "wards": None, "page": pno}
                continue
            if pending is not None and pending["wards"] is None:
                m = WARD_ONLY_RE.match(line)
                if m and _wards(m.group(1)):
                    pending["wards"] = _wards(m.group(1))
                    continue
            if pending is not None and pending["wards"] is not None:
                rows.append({"borough": borough, "sn": pending["sn"], "wards": pending["wards"],
                             "location_text": line, "page": pending["page"]})
                pending = None
                continue
            # continuation of the previous row's text
            if rows and rows[-1]["borough"] == borough:
                rows[-1]["location_text"] = f"{rows[-1]['location_text']} {line}"
    for r in rows:
        r["location_text"] = re.sub(r"\s+", " ", r["location_text"]).strip()
    return rows


def _next_sn(rows, borough):
    same = [r["sn"] for r in rows if r["borough"] == borough]
    return (max(same) + 1) if same else 1


def _expects_sn(rows, borough, line):
    m = SN_ONLY_RE.match(line)
    return bool(m and borough is not None and int(m.group(1)) == _next_sn(rows, borough))


def check(rows):
    errors = []
    by_b = {}
    for r in rows:
        by_b.setdefault(r["borough"], []).append(r["sn"])
    for b in range(1, 17):
        sns = by_b.get(b, [])
        if not sns:
            errors.append(f"borough {b}: no rows")
        elif sns != list(range(1, len(sns) + 1)):
            errors.append(f"borough {b}: SN sequence broken {sns}")
    for r in rows:
        if len(r["location_text"]) < 3:
            errors.append(f"borough {r['borough']} SN {r['sn']}: empty location")
    return errors, {b: len(v) for b, v in sorted(by_b.items())}


def main():
    rows = parse()
    errors, counts = check(rows)
    print("rows per borough:", counts, "total:", len(rows))
    if errors:
        for e in errors:
            print("ERROR", e)
        return 1
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["borough", "sn", "wards", "location_text", "page", "source"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "wards": ",".join(map(str, r["wards"])), "source": SOURCE})
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
