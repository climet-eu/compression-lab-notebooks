# Compression challenge submissions (challenge_work)

Everything in this directory is self-contained and runs inside the repository's
`.venv` (plus `numba`, installed with `uv pip install numba`).

## Codecs (all published packages, none left in this directory)

| module | class / id | idea |
|---|---|---|
| external | `abs_or_rel` ([numcodecs-abs-or-rel](https://github.com/SF-N/numcodecs-abs-or-rel)) | pointwise "abs OR rel" error bound via a linear/log transform with encode-time verification (pv, w and single-level variables with such requirements). |
| external | `clip` ([numcodecs-clip](https://github.com/SF-N/numcodecs-clip)), `chunked` ([numcodecs-chunked](https://github.com/SF-N/numcodecs-chunked)), `grid_int` ([numcodecs-grid-int](https://github.com/SF-N/numcodecs-grid-int)), `combinators.stack` (numcodecs-combinators) | clipping to data limits, per-chunk encoding, bitwise-lossless integer-grid coding, lossless post-compression. |
| external | `interp_ctx` ([numcodecs-interp-ctx](https://github.com/SF-N/numcodecs-interp-ctx)) | SZ3-style coarse-to-fine interpolation prediction with in-loop quantisation + context mixing; best for smooth fields at low bitrates (u, v, t, z, ...). |
| external | `lon_gradient` ([numcodecs-lon-gradient](https://github.com/SF-N/numcodecs-lon-gradient)) | spatial-gradient challenge: bounds the longitude derivative by compressing the stride-10 differences (2x wider step) with an inner abs-error codec, integrates along residue classes with closure-aware rounding. |
| external | `eb_quantize` ([numcodecs-eb-quantize](https://github.com/SF-N/numcodecs-eb-quantize)), `context_mixing.bitmap` / `.symbols` / `.residuals` ([numcodecs-context-mixing](https://github.com/SF-N/numcodecs-context-mixing)) | error-bounded linear quantisation to integer indices + context-mixing entropy coding of the indices; combined with `pw_ratio` (relative bounds) and `mask.meta` (missing values / exact zeros, bitmap coded with `context_mixing.bitmap`). |
| external | `replace.threshold` ([numcodecs-replace](https://github.com/juntyr/numcodecs-replace), `ThresholdFilterCodec` PR), `zero` with `value` ([numcodecs-zero](https://github.com/juntyr/numcodecs-zero), PR) | dropping negligible values under a mean error budget; (near-)constant reconstructions for very loose requirements. |
| external | `mask.meta` ([numcodecs-mask](https://github.com/juntyr/numcodecs-mask), with the mask-aware codec protocol of [PR #4](https://github.com/juntyr/numcodecs-mask/pull/4)) | masks NaNs / zeros; `eb_quantize`, `interp_ctx` and the context-mixing coders implement the protocol and skip masked values entirely. |

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

`search.py` tries codec families (SPERR pwe/q/bpp, `interp_ctx`,
`eb_quantize` + `context_mixing.*`, pw_ratio log-domain variants, abs-or-rel transform, thresholding of
negligible values for mean bounds, zero/NaN masks, clipping, LZMA post-compression), bisects
the error parameter against a fast mirror of `compression_requirement_checks`, and finally
verifies the winner with the official checker.
