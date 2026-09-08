# Online Laboratory for Data Compression in Climate Science and Meteorology

Welcome to the **Online Laboratory for Data Compression in Climate Science and Meteorology**!

This laboratory consists of several Jupyter notebooks that introduce lossy compression of weather and climate datasets, gives an overview of available compressors, and sets out several compression challenges.

You can read through the notebooks in [book form](https://climet-eu.github.io/compression-lab-notebooks/) or follow along in your own [JupyterLab](https://jupyter.org/) environment.


## Getting Started

### (a) Local installation

First, clone the this repository using git:

```shell
git clone https://github.com/climet-eu/compression-lab-notebooks.git
cd compression-lab-notebooks
```

We provide several example datasets, which are stored in Git Large File Storage. If you have not yet installed `git lfs` on your system, you can find instructions here: <https://git-lfs.com>.
Afterwards, you can download the datasets with:

```shell
git lfs install
git lfs fetch --all
```

We use the `uv` Python package manager. If you have not yet installed `uv` on your system, you can find instructions here: <https://docs.astral.sh/uv/getting-started/installation/>.
Afterwards, you can create a fresh virtual environment and install all dependencies using:

```shell
uv sync
```

Finally, you can run Jupyter Lab using:

```shell
uv run jupyter lab
```

### (b) Setup-free Online Laboratory

Alternatively, you can open the notebooks in the [Online Laboratory for Climate Science and Meteorology](https://docs.climet.eu/lab/) at <https://compression.lab.climet.eu/> and get started right away.

Please note that compression in the Online Laboratory currently only works in recent Firefox and Chrome browsers.

```{warning} Warning: JupyterLite may not work in every web browser
<div style="background-color:var(--tw-prose-invert-body)">
<img src="https://baseline.js.org/features/wasm-multi-memory/static-adaptive.svg" alt="Baseline Status: Multi-memory (WebAssembly)"/>
</div>
```


## Glossary

**Bit Pattern**
: The bits that make up a number, e.g. the 32 bits for a single-precision floating-point number.

**Codec**
: An algorithm that transforms the data from one representation to another (encoding) and back (decoding).

**Compression**
: Reducing the number of bits needed to store some data.

**Compressor**
: A codec that implements compression and decompression.

**Filter**
: A codec that transforms the data to make it more easily compressible without necessarily reducing its byte size directly itself.

**IEEE 754 Floating Point Number**
: A number with dynamic precision $(-1)^{s} \cdot 2^{e-b} \cdot 1.m\ldots$ that is represented by its sign $s$, its binary exponent $e$ with bias $b$, and a mantissa $m\ldots$ that encodes the binary fractional multiplier.

**Lossless Compression**
: Compression that reproduces the original bits exactly during decompression.

**Lossy Compression**
: Compression that may only produce an approximation of the original data during decompression.

**Meta-Compressor**
: A compressor that wraps one or more other compressors and transforms the data they work with. Meta-compressors can be used to combine multiple compressors or to provide extra functionality on top of existing compressors.


## Overview of the provided notebooks

The **Online Laboratory for Data Compression in Climate Science and Meteorology** comes with several Jupyter notebook examples to

1. showcase different compression methods on various weather and climate example datasets
2. allow you to easily and quickly test out compression on *your* data

The following is an overview of all notebooks:

- [`01-compression.ipynb`](01-compression.ipynb): Introduction to compression with `numcodecs`
- [`02-datasets/`](02-datasets/README.md): Example datasets and access via an S3 bucket
  - [`01-preprocessed.ipynb`](02-datasets/01-preprocessed.ipynb): Preprocessed example dataset subsets for quick testing
  - [`02-hplp.ipynb`](02-datasets/02-hplp.ipynb): hplp-experiment dataset
  - [`03-OpenIFS.ipynb`](02-datasets/03-OpenIFS.ipynb): OpenIFS dataset
  - [`04-NextGEMS.ipynb`](02-datasets/04-NextGEMS.ipynb): NextGEMS dataset
  - [`05-ICONXPP.ipynb`](02-datasets/05-ICONXPP.ipynb): ICON-XPP dataset
- [`03-compressors/`](03-compressors/README.md): Overview of popular compressors
  - [`01-bit-round.ipynb`](03-compressors/01-bit-round.ipynb): Bit rounding
  - [`02-zfp.ipynb`](03-compressors/02-zfp.ipynb): ZFP
  - [`03-sperr.ipynb`](03-compressors/03-sperr.ipynb): SPERR
  - [`04-ebcc.ipynb`](03-compressors/04-ebcc.ipynb): EBCC
  - [`05-lc.ipynb`](03-compressors/05-lc.ipynb): LC
  - [`06-sz3.ipynb`](03-compressors/06-sz3.ipynb): SZ3
  - [`07-pressio.ipynb`](03-compressors/07-pressio.ipynb): LibPressio
  - [`08-safeguards.ipynb`](03-compressors/08-safeguards.ipynb): Compression Safeguards
  - [`09-random-projection.ipynb`](03-compressors/09-random-projection.ipynb): Random Projection
- [`04-challenges/`](04-challenges/README.md): Compression challenges
  - [`01-nan-missing-values.ipynb`](04-challenges/01-nan-missing-values.ipynb): NaN missing values
  - [`02-relative-error-bound.ipynb`](04-challenges/02-relative-error-bound.ipynb): Pointwise relative error bound
  - [`03-spatial-gradient.ipynb`](04-challenges/03-spatial-gradient.ipynb): Spatial gradient value along the longitude axis
- `05-appendices/`
  - [`01-data-sources/`](05-appendices/01-data-sources/README.md): Opening datasets from different sources
    - [`01-local.ipynb`](05-appendices/01-data-sources/01-local.ipynb): Open a large local read-only dataset
    - [`02-remote.ipynb`](05-appendices/01-data-sources/02-remote.ipynb): Open large remote datasets using `fsspec`, `kerchunk`, and `zarr`
    - [`03-cdsapi.ipynb`](05-appendices/01-data-sources/03-cdsapi.ipynb): Download small datasets from the Climate Data Store using the `cdsapi`
    - [`04-ecmwfapi.ipynb`](05-appendices/01-data-sources/04-ecmwfapi.ipynb): Download small datasets from the ECMWF Archive using the `ecmwfapi`


## Getting Help and Contributing

This laboratory is being developed at https://github.com/climet-eu/lab and https://github.com/climet-eu/compression-lab-notebooks. If you come across a bug or would like to suggest a new feature or support for an additional Python package, please submit an issue at https://github.com/climet-eu/lab/issues/ or https://github.com/climet-eu/compression-lab-notebooks/issues.


## License

Licensed under the CC BY 4.0 license ([LICENSE](LICENSE.txt) or https://creativecommons.org/licenses/by/4.0/).

This product includes software produced by UChicago Argonne, LLC under Contract No. DE-AC02-06CH11357 with the Department of Energy.

The example datasets in the `data/` folder are licensed separately, please see the `LICENSE.txt` files in the respective subfolders.


## Funding

The Online Laboratory for Data Compression in Climate Science and Meteorology has been developed as part of [ESiWACE3](https://www.esiwace.eu), the third phase of the Centre of Excellence in Simulation of Weather and Climate in Europe.

Funded by the European Union. This work has received funding from the European High Performance Computing Joint Undertaking (JU) under grant agreement No 101093054.
