# Opening datasets from different sources

The **Online Laboratory for Data Compression in Climate Science and Meteorology** is built so you can explore various data compression approaches on different datasets, including your own. Since you can run the notebooks in the [Online Laboratory for Climate Science and Meteorology](https://docs.climet.eu/lab/), which runs entirely inside your web browser which has limited memory and is isolated from your filesystem for security, the laboratory supports several approaches to access both small and large datasets from different sources.


## Overview of the provided notebooks

- [`01-local.ipynb`](01-local.ipynb): Open a large local read-only dataset
- [`02-remote.ipynb`](02-remote.ipynb): Open large remote datasets using `fsspec`, `kerchunk`, and `zarr`
- [`03-cdsapi.ipynb`](03-cdsapi.ipynb): Download small datasets from the Climate Data Store using the `cdsapi`
- [`04-ecmwfapi.ipynb`](04-ecmwfapi.ipynb): Download small datasets from the ECMWF Archive using the `ecmwfapi`


## License

Licensed under the CC BY 4.0 license ([LICENSE](../../LICENSE.txt) or https://creativecommons.org/licenses/by/4.0/).
