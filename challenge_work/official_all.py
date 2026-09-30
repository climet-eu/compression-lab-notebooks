"""Run the official compression_requirement_checks on EVERY timestep for an
all-timestep result (results_all/<lk>__<var>.json), in parallel over timesteps.

Usage: python official_all.py <pressure|single> <var> <nproc>
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "results_all"


def _check(args):
    lk, v, t, cfg = args
        import numcodecs_interp_ctx  # noqa: F401
    from compression_requirement_checks import check_safety_requirements
    from numcodecs.registry import get_codec
    from era5 import load_era5_data
    from search import get_requirements

    reqs = get_requirements(lk, v)
    da = load_era5_data(leveltype=lk, param=v)[v]
    x = np.ascontiguousarray(da.isel(time=t).values)
    codec = get_codec(cfg)
    e = codec.encode(x)
    d = np.asarray(codec.decode(np.frombuffer(bytes(e), np.uint8))).reshape(x.shape)
    ok = bool(check_safety_requirements(original=x, reconstructed=d, requirements=reqs))
    return t, ok, len(e)


if __name__ == "__main__":
    import multiprocessing as mp

    lk, v = sys.argv[1], sys.argv[2]
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    path = OUT / f"{lk}__{v}.json"
    r = json.load(open(path))
    T = r["T"]
    t0 = time.time()
    results = {}
    with mp.get_context("spawn").Pool(nproc, maxtasksperchild=8) as pool:
        for t, ok, size in pool.imap_unordered(_check, [(lk, v, t, r["config"]) for t in range(T)]):
            results[t] = (ok, size)
            print(f"{lk} {v} t={t} official_ok={ok} size={size} ({time.time() - t0:.0f}s)", flush=True)
    oks = [results[t][0] for t in range(T)]
    r["official_ok_all_timesteps"] = all(oks)
    r["official_failed_timesteps"] = [t for t in range(T) if not results[t][0]]
    r["total_out_recheck"] = int(sum(results[t][1] for t in range(T)))
    json.dump(r, open(path, "w"), indent=1, default=str)
    print("RESULT", lk, v, "official ok on all", T, "timesteps:", all(oks), "failed:", r["official_failed_timesteps"], "total_out matches:", r["total_out_recheck"] == r["total_out"])
