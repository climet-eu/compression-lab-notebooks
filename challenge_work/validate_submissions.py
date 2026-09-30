"""
Validate the compression-challenge submissions of the public scoreboard and
render Markdown leaderboards for the GitHub issues.

Workflow: the Google Sheet is the open submission form. This script pulls it,
instantiates every entry whose "Configuration" column is a machine-readable
numcodecs configuration (a dict with an ``id``; only codecs that are installed
and registered with numcodecs are used, no code is executed), compresses the
challenge data with it, checks the challenge's safety requirement exactly as
the notebooks do, recomputes the compression ratio and renders one Markdown
table per challenge that

* contains validated entries only (a summary line reports the others),
* is sorted by compression ratio, and
* shows the top 10 directly and the remaining entries in a ``<details>`` block.

Usage::

    python validate_submissions.py [--sheets NaN,PwRel,Gradient,ERA5-Pressure,ERA5-Single]
                                   [--limit N] [--out leaderboards.md] [--json validated.json]

The ERA5 data is read from ``data/era5/<leveltype>__<var>.npz`` (see
``cache_era5.py``) or loaded from the remote reference dataset; the three small
challenge datasets are downloaded to ``data/`` on first use.
"""

import argparse
import ast
import json
import math
import time
import traceback
import urllib.request
from pathlib import Path

import numpy as np
import openpyxl
from numcodecs.registry import get_codec

HERE = Path(__file__).parent
SHEET_ID = "1hWxSr-A9Z5EOpK0ri9UCXKAgJ-WXeG828PTLzw3ZHNo"
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=xlsx"
BUCKET = "https://object-store.os-api.cci1.ecmwf.int/esiwacebucket"
ERA5_TIME = "2026-07-15T12:00:00"
TOP = 10
TOLERANCE = 0.02  # accepted relative deviation of the recomputed ratio


# ----------------------------------------------------------------------------
# challenge data and checks (mirroring the notebooks in 04-challenges)
# ----------------------------------------------------------------------------
def _download(name: str, path: str) -> Path:
    target = HERE / "data" / name
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{BUCKET}/{path}", target)
    return target


def load_small(challenge: str):
    """Return (DataArray, check(da, da_dec) -> violation fraction)."""
    import xarray as xr

    if challenge == "NaN":
        ds = xr.open_dataset(_download("HOAPS_2020-08_6-hourly.nc", "HOAPS/HOAPS_2020-08_6-hourly.nc"), engine="h5netcdf", decode_timedelta=True)
        da = ds["wvpa"].sel(time=slice("2020-08-01", "2020-08-07"))

        def check(da, da_dec):
            return float(np.mean(xr.where(np.isnan(da), ~np.isnan(da_dec), ~(np.abs(da_dec - da) <= 1))))

        return da, check
    if challenge == "PwRel":
        ds = xr.open_dataset(_download("hplp_sfc_regridded_tp_025deg_steps_228_240.nc", "hplp/hplp_sfc_regridded_tp_025deg_steps_228_240.nc"), engine="h5netcdf", decode_timedelta=True)
        da = ds["tp"]

        def check(da, da_dec):
            return float(np.mean(~(np.abs(da_dec - da) <= (np.abs(da) * 0.01))))

        return da, check
    if challenge == "Gradient":
        ds = xr.open_dataset(_download("NextGEMS_regridded_hus_025deg_steps_44_45.nc", "NextGEMS_EW3_ICON_ngc4008/NextGEMS_regridded_hus_025deg_steps_44_45.nc"), engine="h5netcdf", decode_timedelta=True)
        da = ds["hus"]

        def deriv(a):
            return (a.roll(lon=-5) - a.roll(lon=5)) / (np.mod(a.lon.roll(lon=-5) - a.lon.roll(lon=5), 360))

        def check(da, da_dec):
            return float(np.mean(~(np.abs(deriv(da_dec) - deriv(da)) <= 1e-6)))

        return da, check
    raise ValueError(challenge)


def load_era5(leveltype: str, variable: str) -> np.ndarray:
    cached = HERE / "data" / "era5" / f"{leveltype}__{variable}.npz"
    if cached.exists():
        return np.load(cached)["arr"]
    from era5 import load_era5_data

    return load_era5_data(leveltype=leveltype, param=variable)[variable].sel(time=ERA5_TIME).values


def era5_requirements(leveltype: str, variable: str):
    from compression_recommendations import Recommendations

    return Recommendations.provide.search(markers={"grib-short-name": variable, "level-kind": leveltype})


# ----------------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------------
def parse_config(text):
    """A machine-readable configuration is a dict literal with an ``id``."""
    if not isinstance(text, str) or not text.strip().startswith("{"):
        return None
    try:
        config = ast.literal_eval(text.replace("nan", "float('nan')") if "float(" not in text else text)
    except Exception:
        try:
            config = eval(text, {"__builtins__": {}}, {"nan": float("nan"), "np": np, "float": float, "inf": float("inf")})  # noqa: S307
        except Exception:
            return None
    return config if isinstance(config, dict) and "id" in config else None


def roundtrip(codec, values):
    t = time.perf_counter()
    encoded = codec.encode(values)
    t_enc = time.perf_counter() - t
    nbytes = np.asarray(encoded).nbytes if not isinstance(encoded, (bytes, bytearray)) else len(encoded)
    buf = np.frombuffer(bytes(encoded), np.uint8) if isinstance(encoded, (bytes, bytearray)) else encoded
    t = time.perf_counter()
    decoded = codec.decode(buf)
    t_dec = time.perf_counter() - t
    return np.asarray(decoded).reshape(values.shape), values.nbytes / nbytes, t_enc, t_dec


def validate_small(challenge: str, rows: list[dict], limit=None) -> list[dict]:
    da, check = load_small(challenge)
    results = []
    for row in rows[:limit]:
        config = parse_config(row.get("Configuration"))
        entry = dict(row, status="not reproducible (no machine-readable configuration)", valid=False)
        if config is not None:
            try:
                codec = get_codec(config)
                decoded, cr, t_enc, t_dec = roundtrip(codec, da.values)
                da_dec = da.copy(data=decoded.astype(da.dtype) if decoded.dtype != da.dtype else decoded)
                violations = check(da, da_dec)
                entry.update(ratio=cr, violations=violations, enc_gbps=da.nbytes / t_enc / 1e9, dec_gbps=da.nbytes / t_dec / 1e9)
                if violations > 0:
                    entry["status"] = f"requirement violated ({100 * violations:.3g} % of the values)"
                elif abs(cr / float(row["Compression"]) - 1) > TOLERANCE:
                    entry["status"] = f"ratio differs (reported {float(row['Compression']):.2f}, measured {cr:.2f})"
                    entry["valid"] = True
                else:
                    entry["status"] = "validated"
                    entry["valid"] = True
            except Exception as ex:
                entry["status"] = f"failed: {type(ex).__name__}: {str(ex)[:80]}"
        results.append(entry)
        print(challenge, row.get("Author"), entry["status"], flush=True)
    return results


def validate_era5(leveltype: str, rows: list[dict], limit=None) -> list[dict]:
    from compression_requirement_checks import check_safety_requirements

    results = []
    cache: dict = {}
    for row in rows[:limit]:
        variable = str(row.get("Variable", "")).strip()
        timesteps = str(row.get("Default/all timestep(s)", "default")).strip()
        config = parse_config(row.get("Configuration"))
        entry = dict(row, status="not reproducible (no machine-readable configuration)", valid=False)
        if config is None or not variable:
            results.append(entry)
            continue
        if not timesteps.startswith("default"):
            entry["status"] = "not validated (only the default timestep is validated here)"
            results.append(entry)
            continue
        try:
            if variable not in cache:
                cache[variable] = (load_era5(leveltype, variable), era5_requirements(leveltype, variable))
            values, requirements = cache[variable]
            codec = get_codec(config)
            decoded, cr, t_enc, t_dec = roundtrip(codec, values)
            ok = bool(check_safety_requirements(original=values, reconstructed=decoded, requirements=requirements))
            entry.update(ratio=cr, enc_gbps=values.nbytes / t_enc / 1e9, dec_gbps=values.nbytes / t_dec / 1e9)
            if not ok:
                entry["status"] = "safety requirements violated"
            elif abs(cr / float(row["Compression"]) - 1) > TOLERANCE:
                entry["status"] = f"ratio differs (reported {float(row['Compression']):.2f}, measured {cr:.2f})"
                entry["valid"] = True
            else:
                entry["status"] = "validated"
                entry["valid"] = True
        except KeyError:
            entry["status"] = "no safety requirements available for this variable"
        except Exception as ex:
            entry["status"] = f"failed: {type(ex).__name__}: {str(ex)[:80]}"
        results.append(entry)
        print(leveltype, variable, row.get("Author"), entry["status"], flush=True)
    return results


# ----------------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------------
def render_table(title: str, entries: list[dict], group_key=None) -> str:
    valid = [e for e in entries if e.get("valid")]
    invalid = [e for e in entries if not e.get("valid")]
    out = [f"### {title}", ""]
    out.append(f"{len(valid)} validated of {len(entries)} entries" + (f" ({len(invalid)} not validated: " + ", ".join(sorted({e['status'].split(' (')[0] for e in invalid})) + ")" if invalid else "") + ".")
    out.append("")
    if group_key:
        groups: dict = {}
        for e in valid:
            groups.setdefault(e[group_key], []).append(e)
        out.append("| Variable | Compression ratio | Author | Configuration |")
        out.append("|---|---:|---|---|")
        for key in sorted(groups):
            best = max(groups[key], key=lambda e: e["ratio"])
            out.append(f"| {key} | {best['ratio']:.2f} | {best.get('Author', '')} | {best.get('Config Short', '')} |")
        out.append("")
        out.append("<details><summary>All validated entries</summary>")
        out.append("")
        out.append("| Variable | Compression ratio | Author | Configuration |")
        out.append("|---|---:|---|---|")
        for key in sorted(groups):
            for e in sorted(groups[key], key=lambda e: -e["ratio"]):
                out.append(f"| {key} | {e['ratio']:.2f} | {e.get('Author', '')} | {e.get('Config Short', '')} |")
        out.append("")
        out.append("</details>")
        return "\n".join(out)
    ranked = sorted(valid, key=lambda e: -e["ratio"])
    header = ["| # | Compression ratio | Author | Configuration | Compression [GB/s] | Decompression [GB/s] |", "|---:|---:|---|---|---:|---:|"]
    lines = [f"| {i + 1} | {e['ratio']:.2f} | {e.get('Author', '')} | {e.get('Config Short', '')} | {e['enc_gbps']:.4f} | {e['dec_gbps']:.4f} |" for i, e in enumerate(ranked)]
    out += header + lines[:TOP]
    if len(lines) > TOP:
        out += ["", f"<details><summary>{len(lines) - TOP} more validated entries</summary>", ""] + header + lines[TOP:] + ["", "</details>"]
    return "\n".join(out)


def read_sheet(path: Path, name: str) -> list[dict]:
    ws = openpyxl.load_workbook(path, data_only=True)[name]
    rows = list(ws.iter_rows(values_only=True))
    header = [str(h) if h is not None else "" for h in rows[0]]
    out = []
    for r in rows[1:]:
        d = {h: v for h, v in zip(header, r) if h}
        if d.get("Compression") is None:
            continue
        try:
            float(d["Compression"])
        except (TypeError, ValueError):
            continue
        out.append(d)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sheets", default="NaN,PwRel,Gradient,ERA5-Pressure,ERA5-Single")
    parser.add_argument("--limit", type=int, default=None, help="validate at most N entries per sheet")
    parser.add_argument("--out", default="leaderboards.md")
    parser.add_argument("--json", default="validated.json")
    parser.add_argument("--xlsx", default=None, help="use a local copy of the sheet instead of downloading it")
    args = parser.parse_args()

    xlsx = Path(args.xlsx) if args.xlsx else HERE / "scoreboard.xlsx"
    if not args.xlsx:
        urllib.request.urlretrieve(SHEET_URL, xlsx)

    sections, all_results = [], {}
    for sheet in args.sheets.split(","):
        rows = read_sheet(xlsx, sheet)
        try:
            if sheet in ("NaN", "PwRel", "Gradient"):
                results = validate_small(sheet, rows, args.limit)
                sections.append(render_table(f"Challenge: {sheet}", results))
            else:
                leveltype = "pressure" if sheet == "ERA5-Pressure" else "single"
                results = validate_era5(leveltype, rows, args.limit)
                sections.append(render_table(f"Challenge: {sheet} (best validated entry per variable)", results, group_key="Variable"))
        except Exception:
            traceback.print_exc()
            continue
        all_results[sheet] = results
    Path(args.out).write_text("\n\n".join(sections) + "\n")
    json.dump(all_results, open(args.json, "w"), indent=1, default=str)
    print("wrote", args.out, "and", args.json)


if __name__ == "__main__":
    main()
