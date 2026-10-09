import numpy as np, xarray as xr
BASE_URL = "https://object-store.os-api.cci1.ecmwf.int/esiwacebucket"
import json
from functools import cache, lru_cache
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests

ERA5_SOURCES = {
    "pressure": {
        "ref_url": f"{BASE_URL}/ERA5_07_2026/era5_pressure_50_500_850_1000_202607_3h.grib.ref",
        "catalog_url": f"{BASE_URL}/ERA5_07_2026/era5_pressure_50_500_850_1000_202607_3h.grib.refs.json",
    },
    "single": {
        "catalog_url": f"{BASE_URL}/ERA5_07_2026/era5_single_202607_6h.grib.refs.json",
    },
}


def load_era5_data(
    leveltype=None,
    param=None,
    time=None,
    level=None,
    area=None,
    catalog_url=None,
    catalog_path=None,
    ref_url=None,
    ref_path=None,
):
    if param is None:
        raise ValueError("Please specify a GRIB short name or list of short names.")

    params = _as_list(param)
    datasets = []

    if ref_url is None and ref_path is None and leveltype in ERA5_SOURCES:
        ref_url = ERA5_SOURCES[leveltype].get("ref_url")

    if ref_url is not None or ref_path is not None:
        ds = _open_ref(ref_path or ref_url)
        datasets.append(_subset(ds, params, time, level, area))
    else:
        catalog = load_era5_catalog(leveltype, catalog_url, catalog_path)
        catalog_url = catalog_url or ERA5_SOURCES[leveltype]["catalog_url"]
        for group, group_params in _groups_for_params(catalog, params):
            ref_location = _ref_location(group["ref"], catalog_url, catalog_path)
            datasets.append(
                _subset(_open_ref(ref_location), group_params, time, level, area)
            )

    datasets = [ds for ds in datasets if ds.data_vars]
    if not datasets:
        raise ValueError("No variables matched the requested selection.")
    return (
        datasets[0]
        if len(datasets) == 1
        else xr.merge(datasets, join="outer", compat="override")
    )


def _as_list(value):
    if value is None or isinstance(value, (str, bytes)):
        return [value]
    return list(value)


def _load_json(location):
    if str(location).startswith(("http://", "https://")):
        response = requests.get(location)
        response.raise_for_status()
        return response.json()
    return json.loads(Path(location).read_text())


@cache
def _remote_json(location):
    return _load_json(location)


def _ref_location(ref_name, catalog_url=None, catalog_path=None):
    if str(ref_name).startswith(("http://", "https://")):
        return ref_name
    if catalog_path is not None:
        return Path(catalog_path).parent / ref_name
    return urljoin(catalog_url, ref_name)


def _var_name(ds, param):
    aliases = {"2t": "t2m", "2d": "d2m", "10u": "u10", "10v": "v10"}
    for name in (param, aliases.get(param)):
        if name in ds.data_vars:
            return name
    return None


def _open_ref(ref_location):
    ref = _load_json(ref_location)
    return xr.open_dataset(
        "reference://",
        engine="zarr",
        backend_kwargs={
            "storage_options": {
                "fo": ref,
                "asynchronous": True,
                "remote_options": {"asynchronous": True},
            }
        },
        consolidated=False,
        chunks={},
    )


def _subset(ds, param=None, time=None, level=None, area=None):
    if param is not None:
        params = _as_list(param)
        names = [_var_name(ds, name) for name in params]
        missing = [param for param, name in zip(params, names) if name is None]
        if missing:
            raise KeyError(
                f"Variable(s) {missing} not found in this reference group. "
                f"Available variables: {list(ds.data_vars)}"
            )
        ds = ds[names]

    if time is not None and "time" in ds.coords:
        ds = ds.sel(time=[pd.Timestamp(t) for t in _as_list(time)])
    if level is not None and "level" in ds.coords:
        ds = ds.sel(level=level)

    if area is not None:
        north, west, south, east = area
        lon_name = "longitude" if "longitude" in ds.coords else "lon"
        lat_name = "latitude" if "latitude" in ds.coords else "lat"
        lon = ds[lon_name]

        if float(lon.min()) < 0:
            west = west - 360 if west > 180 else west
            east = east - 360 if east > 180 else east
        else:
            west = west % 360
            east = east % 360

        lat_mask = (ds[lat_name] >= south) & (ds[lat_name] <= north)
        if west <= east:
            lon_mask = (lon >= west) & (lon <= east)
        else:
            lon_mask = (lon >= west) | (lon <= east)
        ds = ds.where(lat_mask & lon_mask, drop=True).sortby(lon_name)
    return ds


def _groups_for_params(catalog, params):
    matches = []
    missing = set(params)
    for group in catalog["groups"].values():
        variables = set(group.get("variables", []))
        group_params = [param for param in params if param in variables]
        if group_params:
            matches.append((group, group_params))
            missing -= set(group_params)
    if missing:
        available = sorted(
            {v for g in catalog["groups"].values() for v in g.get("variables", [])}
        )
        raise ValueError(
            f"Parameters not found: {sorted(missing)}. Available: {available}"
        )
    return matches


def load_era5_catalog(leveltype=None, catalog_url=None, catalog_path=None):
    if catalog_path is not None:
        return _load_json(catalog_path)
    if catalog_url is None:
        if leveltype not in ERA5_SOURCES:
            raise ValueError("Use leveltype='pressure' or leveltype='single'.")
        catalog_url = ERA5_SOURCES[leveltype]["catalog_url"]
    return _remote_json(catalog_url)


def era5_data_array(ds, param):
    name = _var_name(ds, param)
    if name is None:
        raise KeyError(
            f"Variable {param!r} not found. Available variables: {list(ds.data_vars)}"
        )
    return ds[name]
