"""Download the public scoreboard (Google Sheet) and extract the previous best
compression ratio per challenge variable into scoreboard_best.json."""

import json
import urllib.request
from pathlib import Path

import openpyxl

SHEET_ID = "1hWxSr-A9Z5EOpK0ri9UCXKAgJ-WXeG828PTLzw3ZHNo"
URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=xlsx"
HERE = Path(__file__).parent


def main():
    xlsx = HERE / "scoreboard.xlsx"
    urllib.request.urlretrieve(URL, xlsx)
    wb = openpyxl.load_workbook(xlsx, data_only=True)
    best = {}
    # (sheet, leveltype, variable column, config-short column, timestep column or None)
    for sheet, lk, vcol, ccol, tcol in (("ERA5-Single", "single", 2, 4, None), ("ERA5-Pressure", "pressure", 3, 5, 4)):
        for r in list(wb[sheet].iter_rows(values_only=True))[1:]:
            if r[1] is None or r[vcol] is None:
                continue
            try:
                cr = float(r[1])
            except (TypeError, ValueError):
                continue
            if tcol is not None and str(r[tcol]).strip() != "default":
                continue
            key = f"{lk}__{str(r[vcol]).strip()}"
            if key not in best or cr > best[key][0]:
                best[key] = (cr, r[0], r[ccol])
    json.dump(best, open(HERE / "scoreboard_best.json", "w"), indent=1)
    xlsx.unlink()
    print("wrote scoreboard_best.json with", len(best), "entries")


if __name__ == "__main__":
    main()
