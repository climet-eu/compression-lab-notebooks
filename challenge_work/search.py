"""Per-variable compressor search for the ERA5 challenges."""

import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from numcodecs.registry import get_codec

import numcodecs_chunked, numcodecs_clip, numcodecs_grid_int  # noqa: F401,E401
import numcodecs_abs_or_rel, numcodecs_context_mixing, numcodecs_eb_quantize, numcodecs_mask, numcodecs_replace, numcodecs_zero  # noqa: F401,E401
import numcodecs_interp_ctx  # noqa: F401
from reqs import analyse, fast_check

HERE = Path(__file__).parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)


def load_var(lk, v):
    return np.load(HERE / "data" / "era5" / f"{lk}__{v}.npz")["arr"]


def get_requirements(lk, v):
    from compression_recommendations import Recommendations

    return Recommendations.provide.search(markers={"grib-short-name": v, "level-kind": lk})


# ----------------------------------------------------------------------------
# codec families
# ----------------------------------------------------------------------------
def _f(p):
    # keep pw_ratio's "$eb_abs" marker strings untouched
    return p if isinstance(p, str) else float(p)


def _sperr_pwe(p):
    return {"id": "sperr.rs", "mode": "pwe", "pwe": _f(p)}


def _sperr_q(p):
    return {"id": "sperr.rs", "mode": "q", "q": float(p)}


def _sz3_abs(p):
    return {"id": "sz3.rs", "eb_mode": "abs", "eb_abs": _f(p), "predictor": "interpolation-lorenzo"}


def _zfp_acc(p):
    return {"id": "zfp.rs", "mode": "fixed-accuracy", "tolerance": float(p)}


def _interp_abs(p):
    return {"id": "interp_ctx", "eb": _f(p)}


def _ebq_residuals(p):
    return {"id": "eb_quantize", "eb": _f(p), "codec": {"id": "context_mixing.residuals"}}


def _ebq_symbols(p):
    return {"id": "eb_quantize", "eb": _f(p), "codec": {"id": "context_mixing.symbols"}}


def _pwr(inner_fn):
    def make(p):
        inner = inner_fn("$eb_abs")
        return {
            "id": "pw_ratio",
            "eb_ratio": 1.0 + float(p),
            "eb_abs_marker": "$eb_abs",
            "log_codec": inner,
            "sign_codec": {"id": "zstd.rs", "level": 19},
        }

    return make


def _absrel(inner_fn, a, r):
    def make(p):
        return {"id": "abs_or_rel", "eb_abs": a * p, "eb_rel": r * p, "codec": inner_fn("$eb_abs"), "eb_abs_marker": "$eb_abs"}

    return make


def _sperr_bpp(p):
    return {"id": "sperr.rs", "mode": "bpp", "bpp": float(max(1e-4, 1.0 / p))}


def _sperr_psnr(rng):
    def make(p):
        return {"id": "sperr.rs", "mode": "psnr", "psnr": float(max(1.0, 20 * math.log10(rng / p)))}

    return make


def _sz3_psnr(rng):
    def make(p):
        return {"id": "sz3.rs", "eb_mode": "psnr", "eb_psnr": float(max(1.0, 20 * math.log10(rng / p))), "predictor": "interpolation-lorenzo"}

    return make


def wrap(cfg, info, mask_nan, mask_zero, per_slice=False, threshold=None):
    if per_slice:
        cfg = {"id": "chunked", "codec": cfg, "chunk_shape": [1, "..."]}
    if threshold is not None:
        cfg = {"id": "combinators.stack", "codecs": [{"id": "replace.threshold", "threshold": float(threshold), "replacement": 0}, cfg]}
    if info.minimum is not None or info.maximum is not None:
        cfg = {"id": "combinators.stack", "codecs": [{"id": "clip", "minimum": info.minimum, "maximum": info.maximum}, cfg]}
    if mask_nan or mask_zero:
        aware = cfg.get("id") in ("eb_quantize", "interp_ctx", "mask.meta") or (
            cfg.get("id") == "combinators.stack" and cfg["codecs"][-1].get("id") in ("eb_quantize", "interp_ctx")
        )
        if mask_nan and not aware:
            cfg = {"id": "combinators.stack", "codecs": [{"id": "replace.filter", "replacements": {"nan": "finite_mean"}}, cfg]}
        if mask_zero:
            cfg = {"id": "mask.meta", "mask": 0.0, "bitmap_codec": {"id": "context_mixing.bitmap"}, "codec": cfg}
        if mask_nan:
            cfg = {"id": "mask.meta", "mask": float("nan"), "bitmap_codec": {"id": "context_mixing.bitmap"}, "codec": cfg}
    return cfg


def evaluate(cfg, x, reqs):
    codec = get_codec(cfg)
    e = codec.encode(x)
    d = np.asarray(codec.decode(np.frombuffer(bytes(e), np.uint8)))
    if d.shape != x.shape:
        d = d.reshape(x.shape)
    if d.dtype != x.dtype:
        return False, len(e), None
    return fast_check(x, d, reqs), len(e), d


def search_family(name, make, p0, x, reqs, log, max_evals=22):
    """Expand then bisect on the error-scale parameter p (larger = smaller output)."""
    evals = 0
    cache = {}

    def ev(p):
        nonlocal evals
        if p in cache:
            return cache[p]
        evals += 1
        t = time.time()
        try:
            ok, size, _ = evaluate(make(p), x, reqs)
        except Exception as ex:  # codec failure counts as fail
            log(f"    {name} p={p:.4g} ERROR {type(ex).__name__}: {str(ex)[:120]}")
            ok, size = False, None
        cache[p] = (ok, size)
        log(f"    {name} p={p:.4g} ok={ok} size={size} ({time.time()-t:.1f}s)")
        return ok, size

    best = None  # (size, p)

    def consider(p, ok, size):
        nonlocal best
        if ok and size is not None and (best is None or size < best[0]):
            best = (size, p)

    p = p0
    ok, size = ev(p)
    consider(p, ok, size)
    if ok:
        p_pass = p
        p_fail = None
        for _ in range(16):
            p *= 2.0
            ok, size = ev(p)
            consider(p, ok, size)
            if ok:
                p_pass = p
            else:
                p_fail = p
                break
    else:
        p_fail = p
        p_pass = None
        for _ in range(16):
            p /= 2.0
            ok, size = ev(p)
            consider(p, ok, size)
            if ok:
                p_pass = p
                break
            p_fail = p
    if p_pass is None:
        return None
    if p_fail is not None:
        for _ in range(4):
            if evals >= max_evals:
                break
            pm = math.sqrt(p_pass * p_fail)
            ok, size = ev(pm)
            consider(pm, ok, size)
            if ok:
                p_pass = pm
            else:
                p_fail = pm
    return best


def build_families(x, info, reqs, lk):
    fin = x[np.isfinite(x)]
    has_nan = bool(np.isnan(x).any())
    zero_frac = float((fin == 0).mean()) if fin.size else 0.0
    rng = float(fin.max() - fin.min()) if fin.size else 1.0
    mean_abs_x = float(np.abs(fin).mean()) if fin.size else 1.0
    is3d = x.ndim == 3
    fams = []  # (name, make(p), p0, wrappers-kwargs)

    if info.lossless:
        wk = {"mask_nan": False, "mask_zero": False}
        return [("grid-int/residuals", lambda p: {"id": "grid_int", "codec": {"id": "context_mixing.residuals"}}, 1.0, wk),
                ("grid-int/symbols", lambda p: {"id": "grid_int", "codec": {"id": "context_mixing.symbols"}}, 1.0, wk)], has_nan, zero_frac

    # error scale p0
    abs_bounds = list(info.max_abs) + [v * rng for v in info.max_range_rel]
    mean_bounds = list(info.mean_abs) + [v * mean_abs_x for v in info.mean_rel] + [v * rng for v in info.mean_range_rel]
    pointwise_only_rel = bool(info.max_rel) and not abs_bounds and not info.abs_or_rel
    need_zero_exact = bool(info.mean_rel)

    if abs_bounds:
        p0_abs = max(abs_bounds)
    elif mean_bounds:
        p0_abs = 4.0 * max(mean_bounds)
    else:
        p0_abs = rng * 0.01

    if need_zero_exact:
        zero_variants = [True]
    elif zero_frac > 0.03:
        zero_variants = [False, True]
    else:
        zero_variants = [False]

    def add(name, make, p0, mask_zero, per_slice=False, threshold=None):
        wk = {"mask_nan": has_nan, "mask_zero": mask_zero, "per_slice": per_slice}
        if threshold is not None:
            wk["threshold"] = threshold
        fams.append((name, make, p0, wk))

    # heavy-tail thresholds: zero out the smallest values using a fraction of
    # the mean error budget (only for pure mean bounds)
    thresholds = []
    if info.has_mean and not info.has_pointwise and fin.size:
        n_fin = fin.size
        budgets = [v * n_fin for v in info.mean_abs] + [v * float(np.abs(fin).sum()) for v in info.mean_rel] + [v * rng * n_fin for v in info.mean_range_rel]
        nz = np.sort(np.abs(fin[fin != 0]))
        if nz.size:
            cs = np.cumsum(nz)
            bmin, bmax = min(budgets), max(budgets)
            for budget, fracs in ((bmin, (0.3, 0.7)), (bmax, (0.3,))):
                if budget == bmin and bmax == bmin and fracs == (0.3,):
                    continue
                for frac in fracs:
                    k = int(np.searchsorted(cs, frac * budget))
                    if k >= 0.03 * nz.size and k < nz.size and (frac, float(nz[k])) not in thresholds:
                        thresholds.append((frac, float(nz[k])))

    # data on a uniform grid (GRIB packing): coding exactly at the grid step is
    # lossless -> also try it (and coarser multiples) for mean bounds
    if info.has_mean and not info.has_pointwise:
        try:
            from numcodecs_grid_int import GridIntCodec
            _off, gscale = GridIntCodec.detect(x)
        except Exception:
            gscale = None
        if gscale and gscale > 0 and p0_abs < 2 * gscale:
            def _grid_symbols(p, gs=gscale):
                return {"id": "eb_quantize", "eb": float(p) * gs / 2 * (1 - 1e-9), "codec": {"id": "context_mixing.symbols"}}

            def _grid_residuals(p, gs=gscale):
                return {"id": "eb_quantize", "eb": float(p) * gs / 2 * (1 - 1e-9), "codec": {"id": "context_mixing.residuals"}}

            for mz in zero_variants:
                sfx = "+zeromask" if mz else ""
                add("grid-symbols" + sfx, _grid_symbols, 1.0, mz)
                add("grid-residuals" + sfx, _grid_residuals, 1.0, mz)

    if not pointwise_only_rel:
        for mz in zero_variants:
            sfx = "+zeromask" if mz else ""
            add("interp-abs" + sfx, _interp_abs, p0_abs, mz)
            add("sperr-pwe" + sfx, _sperr_pwe, p0_abs, mz)
            add("ebq-residuals" + sfx, _ebq_residuals, p0_abs, mz)
            if rng / (2 * p0_abs) <= 512:
                add("ebq-symbols" + sfx, _ebq_symbols, p0_abs, mz)
            if info.has_mean and not info.has_pointwise:
                add("sperr-q" + sfx, _sperr_q, p0_abs, mz)
                add("sperr-bpp" + sfx, _sperr_bpp, 1.0, mz)
            for frac, thr in thresholds:
                add(f"interp-abs+thr{frac}" + sfx, _interp_abs, p0_abs, mz, threshold=thr)
                add(f"ebq-residuals+thr{frac}" + sfx, _ebq_residuals, p0_abs, mz, threshold=thr)
                if info.has_mean and not info.has_pointwise:
                    add(f"sperr-q+thr{frac}" + sfx, _sperr_q, p0_abs, mz, threshold=thr)

    if info.max_rel:
        p0_rel = max(info.max_rel)
        add("pwratio-ebq-residuals", _pwr(_ebq_residuals), p0_rel, False)
        add("pwratio-interp", _pwr(_interp_abs), p0_rel, False)
        add("pwratio-sperr", _pwr(_sperr_pwe), p0_rel, False)
    elif info.mean_rel and not info.has_pointwise:
        # log-domain / abs-or-rel coding tuned to the mean-relative budget
        p0_rel = 2.0 * max(info.mean_rel)
        add("pwratio-interp", _pwr(_interp_abs), p0_rel, False)
        for frac, thr in thresholds:
            add(f"pwratio-interp+thr{frac}", _pwr(_interp_abs), p0_rel, False, threshold=thr)
        a = max(info.mean_rel) * mean_abs_x
        r = max(info.mean_rel)
        add("absrel-interp", _absrel(_interp_abs, a, r), 1.0, need_zero_exact)
        add("absrel-ebq", _absrel(_ebq_residuals, a, r), 1.0, need_zero_exact)
        add("absrel-sperr", _absrel(_sperr_pwe, a, r), 1.0, need_zero_exact)
    for a, r in info.abs_or_rel:
        add("absrel-interp", _absrel(_interp_abs, a, r), 1.0, False)
        add("absrel-sperr", _absrel(_sperr_pwe, a, r), 1.0, False)
        add("absrel-ebq", _absrel(_ebq_residuals, a, r), 1.0, False)
    return fams, has_nan, zero_frac


def config_short(name, wk, post):
    parts = []
    if wk.get("threshold") is not None:
        parts.append(f"replace.threshold({wk['threshold']:.3g})")
    if wk.get("mask_nan") or wk.get("mask_zero"):
        m = []
        if wk.get("mask_nan"):
            m.append("NaN")
        if wk.get("mask_zero"):
            m.append("zero")
        parts.append("Mask(" + "/".join(m) + ")")
    parts.append(name)
    if post:
        parts.append("LZMA")
    return " + ".join(parts)


def run_variable(lk, v, force=False):
    out_path = RESULTS / f"{lk}__{v}.json"
    if out_path.exists() and not force:
        return json.load(open(out_path))
    log_path = RESULTS / f"{lk}__{v}.log"
    logf = open(log_path, "w")

    def log(msg):
        logf.write(msg + "\n")
        logf.flush()

    t0 = time.time()
    x = load_var(lk, v)
    try:
        reqs = get_requirements(lk, v)
    except KeyError:
        res = {"leveltype": lk, "variable": v, "error": "no recommendation"}
        json.dump(res, open(out_path, "w"))
        return res
    info = analyse(reqs)
    log(f"{lk} {v} shape={x.shape} reqs={' && '.join(r.humanise() for r in reqs)}")
    fams, has_nan, zero_frac = build_families(x, info, reqs, lk)
    candidates = []
    # constant field candidate
    fin = x[np.isfinite(x)]
    for val in {float(np.median(fin)), float(fin.mean())} if fin.size else set():
        cfg = {"id": "zero", "value": val}
        if has_nan:
            cfg = {"id": "mask.meta", "mask": float("nan"), "bitmap_codec": {"id": "context_mixing.bitmap"}, "codec": cfg}
        try:
            ok, size, _ = evaluate(cfg, x, reqs)
        except Exception:
            ok, size = False, None
        log(f"    constant {val}: ok={ok} size={size}")
        if ok:
            candidates.append((size, cfg, "Constant field", {}))
    for name, make, p0, wk in fams:
        try:
            best = search_family(name, lambda p, make=make, wk=wk: wrap(make(p), info, **wk), p0, x, reqs, log)
        except Exception:
            log(traceback.format_exc())
            best = None
        if best is not None:
            size, p = best
            cfg = wrap(make(p), info, **wk)
            candidates.append((size, cfg, config_short(name, wk, False), {"family": name, "p": p}))
            log(f"  => {name}: best size {size} CR {x.nbytes/size:.2f} at p={p:.6g}")
    if not candidates:
        res = {"leveltype": lk, "variable": v, "error": "no passing candidate"}
        json.dump(res, open(out_path, "w"))
        return res
    candidates.sort(key=lambda c: c[0])
    # refine the parameter of the top candidates (extra bisection steps)
    refined = []
    for size, cfg, short, meta in candidates[:2]:
        if "p" not in meta:
            continue
        fam = [f for f in fams if f[0] == meta["family"]]
        if not fam:
            continue
        name, make, p0, wk = fam[0]
        p_pass = meta["p"]
        p_fail = p_pass * 1.19
        best_size, best_p = size, p_pass
        for _ in range(4):
            pm = math.sqrt(p_pass * p_fail)
            try:
                ok, sz, _ = evaluate(wrap(make(pm), info, **wk), x, reqs)
            except Exception:
                ok, sz = False, None
            log(f"    refine {name} p={pm:.6g} ok={ok} size={sz}")
            if ok:
                p_pass = pm
                if sz < best_size:
                    best_size, best_p = sz, pm
            else:
                p_fail = pm
        if best_p != p_pass or best_size < size:
            refined.append((best_size, wrap(make(best_p), info, **wk), short, {"family": name, "p": best_p}))
            log(f"  refined {name}: {size} -> {best_size} at p={best_p:.6g}")
    candidates = sorted(candidates + refined, key=lambda c: c[0])
    # try lossless post-compression on the best few
    extra = []
    post_opts = [({"id": "lzma", "preset": 9}, "LZMA"), ({"id": "lzma", "preset": 2147483657}, "LZMA-9e"), ({"id": "zstd.rs", "level": 22}, "Zstd-22")]
    for size, cfg, short, meta in candidates[:3]:
        if short.startswith("Constant"):
            continue
        for lcfg, lname in post_opts:
            pcfg = {"id": "combinators.stack", "codecs": [cfg, lcfg]}
            try:
                ok, psize, _ = evaluate(pcfg, x, reqs)
            except Exception:
                ok, psize = False, None
            if ok and psize < size:
                extra.append((psize, pcfg, short + " + " + lname, meta))
                log(f"  post-{lname} on {short}: {size} -> {psize}")
    candidates = sorted(candidates + extra, key=lambda c: c[0])
    # official verification of the best candidates
    from compression_requirement_checks import check_safety_requirements

    final = None
    for size, cfg, short, meta in candidates[:4]:
        codec = get_codec(cfg)
        e = codec.encode(x)
        d = np.asarray(codec.decode(np.frombuffer(bytes(e), np.uint8))).reshape(x.shape)
        t = time.time()
        ok = bool(check_safety_requirements(original=x, reconstructed=d, requirements=reqs))
        log(f"  OFFICIAL check {short}: ok={ok} size={len(e)} ({time.time()-t:.1f}s)")
        if ok:
            final = (len(e), cfg, short, meta)
            break
    res = {
        "leveltype": lk,
        "variable": v,
        "shape": list(x.shape),
        "dtype": x.dtype.str,
        "nbytes": int(x.nbytes),
        "requirements": " && ".join(r.humanise() for r in reqs),
        "has_nan": has_nan,
        "zero_frac": zero_frac,
        "time_s": time.time() - t0,
        "candidates": [{"size": s, "cr": x.nbytes / s, "short": sh, "meta": m} for s, c, sh, m in candidates[:8]],
    }
    if final is not None:
        size, cfg, short, meta = final
        res.update({"size": size, "cr": x.nbytes / size, "config": cfg, "config_short": short, "meta": meta, "official_ok": True})
    else:
        res.update({"official_ok": False})
    json.dump(res, open(out_path, "w"), indent=1, default=str)
    logf.close()
    return res


def _worker(args):
    lk, v = args
    try:
        r = run_variable(lk, v)
        return (lk, v, r.get("cr"), r.get("config_short"), r.get("error"))
    except Exception as ex:
        traceback.print_exc()
        return (lk, v, None, None, repr(ex))


if __name__ == "__main__":
    import multiprocessing as mp

    lk = sys.argv[1]
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    only = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    files = sorted((HERE / "data" / "era5").glob(f"{lk}__*.npz"))
    variables = [f.stem.split("__", 1)[1] for f in files]
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
