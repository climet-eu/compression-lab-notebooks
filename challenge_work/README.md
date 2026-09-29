# Compression challenge submissions (challenge_work)

Everything in this directory is self-contained and runs inside the repository's
`.venv` (plus `numba`, installed with `uv pip install numba`).

## Custom codecs (numcodecs API, registered ids)

| module | class / id | idea |
|---|---|---|
| `ctxcoder.py` | `NanContextCodec` / `nan-context-mixing` | uniform quantisation (bin `2*eb`) + NaN mask; quantisation indices coded directly with a context-mixing binary arithmetic coder (neighbour values as contexts). Best for small alphabets / noisy data. |
| `ctxcodec2.py` | `CtxCodec` / `ctx-mixing` | prediction residuals (MED / previous-slice predictor, adaptively selected) coded with context mixing; `mode="abs"` (absolute bound) or `mode="rel"` (log-domain, pointwise relative bound, exact zeros, sign plane). |
| `interpcodec.py` | `InterpCtxCodec` / `interp-ctx` | SZ3-style coarse-to-fine cubic/linear interpolation prediction (in-loop quantisation, `|x-x_dec|<=eb`) + context-mixing coder. Best for smooth fields at low bitrates. |
| `gradcodec.py` | `LonGradientCodec` / `lon-gradient-difference` | for the spatial-gradient challenge: compresses the stride-10 longitude differences (the exact quantity that is bounded) with a 2x wider step, integrates along residue classes, closure-aware rounding + ramp. |
| `wrappers.py` | `mask-fill`, `clip`, `abs-or-rel-transform`, `grid-int`, `per-slice`, `post-lossless`, `threshold-to-zero`, `constant-field` | wrappers used by the ERA5 submissions (NaN/zero masks coded with the context-mixing mask coder, clipping to data limits, transform for "abs OR rel" bounds, exact lossless integer-grid coding, ...). |

The custom codec ids are only known to numcodecs once this package is installed (it registers
them as `numcodecs.codecs` entry points):

```bash
uv pip install -e ./challenge_work      # from the repository root; re-run after `uv sync`
```

Afterwards `numcodecs.registry.get_codec(config)` works for every "Configuration" string in the
Excel files without any import. Alternatively `sys.path.insert(0, "../challenge_work")` and
`import wrappers, interpcodec, gradcodec` registers them as well. The last Excel column
("Codec (paste into the notebook's compressor cell)") contains fully explicit Python
(imports + nested constructor calls) that reproduces each entry.

## Reproduction

Only the code is versioned; data, per-variable results and the Excel files are outputs
(ignored by git) and are regenerated with:

```bash
python cache_era5.py                 # cache ERA5 variables (default timestep) as data/era5/*.npz
python fetch_scoreboard.py           # previous best per variable from the Google Sheet -> scoreboard_best.json
python search.py single 6            # per-variable search (results/*.json), several hours
python search.py pressure 2
python finalize.py 4                 # official re-verification + timing -> submissions/*.xlsx (also NaN/PwRel/Gradient)
# optional: all-timestep evaluation of single variables, each timestep compressed independently
python alltime.py pressure 1 u && python official_all.py pressure u 5 && python add_alltime_rows.py && python update_alltime_notes.py
```

The three small challenges read their datasets from `data/` (downloaded from the ESiWACE bucket
as in the notebooks); `cache_era5.py` does not fetch those, see `finalize.py` for the file names.

`search.py` tries codec families (SPERR pwe/q/bpp, `interp-ctx`, `ctx-mixing`,
`nan-context-mixing`, pw_ratio log-domain variants, abs-or-rel transform, thresholding of
negligible values for mean bounds, zero/NaN masks, clipping, LZMA post-compression), bisects
the error parameter against a fast mirror of `compression_requirement_checks`, and finally
verifies the winner with the official checker.
