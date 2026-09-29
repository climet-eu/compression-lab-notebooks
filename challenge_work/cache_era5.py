"""Cache all ERA5 challenge variables (single timestep) as .npy files."""

import sys
import traceback
from pathlib import Path

import numpy as np

from era5 import load_era5_catalog, load_era5_data

TIME = "2026-07-15T12:00:00"
OUT = Path(__file__).parent / "data" / "era5"
OUT.mkdir(parents=True, exist_ok=True)


def main(leveltypes):
    for lk in leveltypes:
        catalog = load_era5_catalog(leveltype=lk)
        variables = sorted(
            {v for group in catalog["groups"].values() for v in group["variables"]}
        )
        for v in variables:
            path = OUT / f"{lk}__{v}.npz"
            if path.exists():
                continue
            try:
                ds = load_era5_data(leveltype=lk, param=v)
                da = ds[v].sel(time=TIME)
                arr = da.values
                np.savez_compressed(path, arr=arr)
                print(lk, v, arr.shape, arr.dtype, flush=True)
            except Exception:
                print("FAILED", lk, v, flush=True)
                traceback.print_exc()


if __name__ == "__main__":
    main(sys.argv[1:] or ["pressure", "single"])
