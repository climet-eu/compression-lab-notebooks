# Compression challenge submissions (challenge_work)

Everything in this directory is self-contained and runs inside the repository's
`.venv` (plus `numba`, installed with `uv pip install numba`).

## Custom codecs (numcodecs API, registered ids)

| module | class / id | idea |
|---|---|---|
| `ctxcoder.py` | `NanContextCodec` / `nan-context-mixing` | uniform quantisation (bin `2*eb`) + NaN mask; quantisation indices coded directly with a context-mixing binary arithmetic coder (neighbour values as contexts). Best for small alphabets / noisy data. |
| `ctxcodec2.py` | `CtxCodec` / `ctx-mixing` | prediction residuals (MED / previous-slice predictor, adaptively selected) coded with context mixing; `mode="abs"` (absolute bound) or `mode="rel"` (log-domain, pointwise relative bound, exact zeros, sign plane). |
| `wrappers.py` | `mask-fill`, `abs-or-rel-transform`, `threshold-to-zero`, `constant-field` | remaining wrappers used by the ERA5 submissions (NaN/zero masks coded with the context-mixing mask coder, transform for "abs OR rel" bounds, dropping negligible values, constant fields). |
| external | `clip` ([numcodecs-clip](https://github.com/SF-N/numcodecs-clip)), `chunked` ([numcodecs-chunked](https://github.com/SF-N/numcodecs-chunked)), `grid_int` ([numcodecs-grid-int](https://github.com/SF-N/numcodecs-grid-int)), `combinators.stack` (numcodecs-combinators) | clipping to data limits, per-chunk encoding, bitwise-lossless integer-grid coding, lossless post-compression. |
| external | `interp_ctx` ([numcodecs-interp-ctx](https://github.com/SF-N/numcodecs-interp-ctx)) | SZ3-style coarse-to-fine interpolation prediction with in-loop quantisation + context mixing; best for smooth fields at low bitrates (u, v, t, z, ...). |
| external | `lon_gradient` ([numcodecs-lon-gradient](https://github.com/SF-N/numcodecs-lon-gradient)) | spatial-gradient challenge: bounds the longitude derivative by compressing the stride-10 differences (2x wider step) with an inner abs-error codec, integrates along residue classes with closure-aware rounding. |
| external | `eb_quantize` ([numcodecs-eb-quantize](https://github.com/SF-N/numcodecs-eb-quantize)), `context_mixing.bitmap` / `.symbols` / `.residuals` ([numcodecs-context-mixing](https://github.com/SF-N/numcodecs-context-mixing)) | the modular form of `ctx-mixing` / `nan-context-mixing`: error-bounded linear quantisation to integer indices + context-mixing entropy coding of the indices; combined with `pw_ratio` (relative bounds) and `mask.meta` / `replace.filter` (missing values). |

The monolithic `ctx-mixing` and `nan-context-mixing` codecs are kept because `mask.meta` cannot
yet tell the inner codec which positions are masked: the modular composition codes the filled
missing values too (identical ratios where there are no masks, -1% for log-domain coding via
`pw_ratio`, but -13% for the 69%-NaN missing-values challenge). A mask-aware codec protocol for
`numcodecs-mask` is planned.

The custom codec ids are only known to numcodecs once this package is installed (it registers
them as `numcodecs.codecs` entry points):

```bash
uv pip install -e ./challenge_work      # from the repository root; re-run after `uv sync`
# plus the separately published codecs (until they are on PyPI, install from their repos):
uv pip install -e ../numcodecs-clip -e ../numcodecs-chunked -e ../numcodecs-grid-int \
               -e ../numcodecs-eb-quantize -e ../numcodecs-context-mixing \
               -e ../numcodecs-interp-ctx -e ../numcodecs-lon-gradient
```

Afterwards `numcodecs.registry.get_codec(config)` works for every "Configuration" string in the
Excel files without any import. Alternatively `sys.path.insert(0, "../challenge_work")` and
`import wrappers` registers them as well. The last Excel column
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

`search.py` tries codec families (SPERR pwe/q/bpp, `interp_ctx`, `ctx-mixing`,
`nan-context-mixing`, `eb_quantize` + `context_mixing.*`, pw_ratio log-domain variants, abs-or-rel transform, thresholding of
negligible values for mean bounds, zero/NaN masks, clipping, LZMA post-compression), bisects
the error parameter against a fast mirror of `compression_requirement_checks`, and finally
verifies the winner with the official checker.
