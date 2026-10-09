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

## Validating submissions (proposed workflow)

The Google Sheet is the open submission form. `validate_submissions.py` pulls it, instantiates
every entry whose "Configuration" is a machine-readable numcodecs configuration (registered codecs
only, no code execution), compresses the challenge data, checks the safety requirement exactly as
the notebooks do, recomputes the ratio and renders one Markdown leaderboard per challenge with
validated entries only (top 10 visible, the rest in a `<details>` block) for the GitHub issues:

```bash
python validate_submissions.py --sheets NaN,PwRel,Gradient,ERA5-Pressure,ERA5-Single --out leaderboards.md
```

## Search pipeline

The scripts that search for codec configurations and build the Excel submission files
(`search.py`, `finalize.py`, `alltime.py`, ...) live in the private repository
[SF-N/compression-challenge-search](https://github.com/SF-N/compression-challenge-search);
`era5.py` (ERA5 reference loading, copied from the challenge notebooks) is shared.
