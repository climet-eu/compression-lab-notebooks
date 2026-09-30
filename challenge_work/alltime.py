"""
All-timestep evaluation: for every variable, find ONE configuration (starting
from the default-timestep winner) whose reconstruction satisfies the safety
requirements on EVERY timestep, each timestep being compressed independently.

Pass 1 streams through the timesteps and tightens the error parameter whenever
a timestep fails.  If the parameter changed, pass 2 re-evaluates all timesteps
with the final configuration.  Requirements are checked per timestep with the
fast mirror of compression_requirement_checks (conservative margin); the
official checker is additionally run on sampled timesteps.

Usage: python alltime.py <pressure|single> <nproc> [var,var,...] [reverse]
"""

import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

import numcodecs_interp_ctx  # noqa: F401
from era5 import load_era5_data
from reqs import analyse, fast_check
from search import RESULTS, build_families, evaluate, get_requirements, load_var, wrap

HERE = Path(__file__).parent
OUT = HERE / "results_all"
OUT.mkdir(exist_ok=True)
SHRINK = 0.85


def _timed(cfg, x, reqs):
    from numcodecs.registry import get_codec

    codec = get_codec(cfg)
    t = time.perf_counter()
    e = codec.encode(x)
    te = time.perf_counter() - t
    buf = np.frombuffer(bytes(e), np.uint8)
    t = time.perf_counter()
    d = np.asarray(codec.decode(buf)).reshape(x.shape)
    td = time.perf_counter() - t
    ok = d.dtype == x.dtype and fast_check(x, d, reqs)
    return ok, len(e), d, te, td


def run_variable(lk, v):
    out_path = OUT / f"{lk}__{v}.json"
    if out_path.exists():
        return json.load(open(out_path))
    logf = open(OUT / f"{lk}__{v}.log", "w")

    def log(msg):
        logf.write(msg + "\n")
        logf.flush()

    t0 = time.time()
    res = json.load(open(RESULTS / f"{lk}__{v}.json"))
    if "config" not in res:
        r = {"leveltype": lk, "variable": v, "error": res.get("error", "no default result")}
        json.dump(r, open(out_path, "w"))
        return r
    reqs = get_requirements(lk, v)
    info = analyse(reqs)
    x0 = load_var(lk, v)
    fams, has_nan, zero_frac = build_families(x0, info, reqs, lk)
    fam_by_name = {name: (make, wk) for name, make, p0, wk in fams}

    # candidate list: winner first, then the other default-timestep candidates
    cands = []
    meta = res.get("meta", {})
    if meta.get("family") in fam_by_name:
        cands.append((meta["family"], meta["p"], res["config"]))
    else:
        cands.append((None, None, res["config"]))  # e.g. constant field
    for c in res.get("candidates", []):
        m = c.get("meta", {})
        if m.get("family") in fam_by_name and (m["family"], m["p"]) not in [(a, b) for a, b, _ in cands]:
            cands.append((m["family"], m["p"], None))
    cfg0 = res["config"]
    post = None
    if cfg0.get("id") == "combinators.stack" and len(cfg0["codecs"]) == 2 and cfg0["codecs"][1].get("id") in ("lzma", "zstd.rs"):
        post = cfg0["codecs"][1]

    def cfg_for(fam, p):
        make, wk = fam_by_name[fam]
        cfg = wrap(make(p), info, **wk)
        if post:
            cfg = {"id": "combinators.stack", "codecs": [cfg, post]}
        return cfg

    ds = load_era5_data(leveltype=lk, param=v)
    da = ds[v]
    T = int(da.sizes["time"])
    log(f"{lk} {v}: T={T} shape={tuple(da.shape)} reqs={' && '.join(r.humanise() for r in reqs)}")

    # ---------------- pass 1: find a configuration valid on every timestep
    ci = 0
    fam, p, cfg = cands[0]
    if cfg is None:
        cfg = cfg_for(fam, p)
    changed = False
    sizes = []
    times = []
    t = 0
    while t < T:
        x = np.ascontiguousarray(da.isel(time=t).values)
        ok, size, d, te, td = _timed(cfg, x, reqs)
        if ok:
            sizes.append(size)
            times.append((te, td))
            if t % 20 == 0:
                log(f"  t={t} ok size={size} CR={x.nbytes / size:.2f} p={p}")
            t += 1
            continue
        # tighten / fall back
        changed = True
        log(f"  t={t} FAIL at p={p} (fam={fam}); tightening")
        fixed = False
        if fam is not None:
            for _ in range(20):
                p *= SHRINK
                cfg = cfg_for(fam, p)
                ok, size, d, te, td = _timed(cfg, x, reqs)
                log(f"    p={p:.6g} ok={ok} size={size}")
                if ok:
                    fixed = True
                    break
        if not fixed:
            ci += 1
            if ci >= len(cands):
                r = {"leveltype": lk, "variable": v, "error": f"no configuration valid on timestep {t}"}
                json.dump(r, open(out_path, "w"))
                return r
            fam, p, cfg = cands[ci]
            if cfg is None:
                cfg = cfg_for(fam, p)
            log(f"  switching to candidate {ci}: {fam} p={p}")
        # re-check this timestep with the new configuration (loop without incrementing t)

    # ---------------- pass 2: full evaluation with the final configuration
    if changed:
        log(f"  final configuration: fam={fam} p={p}; re-evaluating all timesteps")
        sizes, times = [], []
        for t in range(T):
            x = np.ascontiguousarray(da.isel(time=t).values)
            ok, size, d, te, td = _timed(cfg, x, reqs)
            if not ok:
                r = {"leveltype": lk, "variable": v, "error": f"final configuration failed on timestep {t}"}
                json.dump(r, open(out_path, "w"))
                return r
            sizes.append(size)
            times.append((te, td))

    # official checker on sampled timesteps
    from compression_requirement_checks import check_safety_requirements
    from numcodecs.registry import get_codec

    codec = get_codec(cfg)
    official = {}
    for t in sorted({0, T // 2, T - 1}):
        x = np.ascontiguousarray(da.isel(time=t).values)
        e = codec.encode(x)
        d = np.asarray(codec.decode(np.frombuffer(bytes(e), np.uint8))).reshape(x.shape)
        official[t] = bool(check_safety_requirements(original=x, reconstructed=d, requirements=reqs))
    nbytes_step = int(x.nbytes)
    total_in = nbytes_step * T
    total_out = int(sum(sizes))
    r = {
        "leveltype": lk,
        "variable": v,
        "T": T,
        "shape_step": list(x.shape),
        "requirements": " && ".join(q.humanise() for q in reqs),
        "config": codec.get_config(),
        "config_short": res["config_short"],
        "family": fam,
        "p": p,
        "p_default": res.get("meta", {}).get("p"),
        "changed": changed,
        "cr_all": total_in / total_out,
        "cr_default_timestep": res["cr"],
        "cr_step_min": nbytes_step / max(sizes),
        "cr_step_max": nbytes_step / min(sizes),
        "total_in": total_in,
        "total_out": total_out,
        "enc_gbps": total_in / sum(a for a, _ in times) / 1e9,
        "dec_gbps": total_in / sum(b for _, b in times) / 1e9,
        "official_ok_samples": official,
        "time_s": time.time() - t0,
    }
    json.dump(r, open(out_path, "w"), indent=1, default=str)
    log(f"DONE CR_all={r['cr_all']:.2f} (default {res['cr']:.2f}) changed={changed} official={official} in {r['time_s']:.0f}s")
    logf.close()
    return r


def _worker(args):
    lk, v = args
    try:
        r = run_variable(lk, v)
        return (lk, v, r.get("cr_all"), r.get("changed"), r.get("error"))
    except Exception as ex:
        traceback.print_exc()
        return (lk, v, None, None, repr(ex))


if __name__ == "__main__":
    import multiprocessing as mp

    lk = sys.argv[1]
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    only = sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[3] != "-" else None
    files = sorted(RESULTS.glob(f"{lk}__*.json"))
    variables = [f.stem.split("__", 1)[1] for f in files if "config" in json.load(open(f))]
    if only:
        variables = [v for v in variables if v in only]
    tasks = [(lk, v) for v in variables]
    if len(sys.argv) > 4 and sys.argv[4] == "reverse":
        tasks = tasks[::-1]
    if nproc == 1:
        for t in tasks:
            print(_worker(t), flush=True)
    else:
        with mp.get_context("spawn").Pool(nproc, maxtasksperchild=1) as pool:
            for r in pool.imap_unordered(_worker, tasks):
                print(r, flush=True)
