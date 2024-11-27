# Opening datasets from different sources

The **Online Laboratory for Data Compression** is built so you can explore various data compression approaches on different datasets, including your own. Since the online lab runs in your web browser, which has limited memory and is isolated from your filesystem for security, the lab supports several approaches to access both small and large datasets from different sources.


## Overview of the provided notebooks

- [`01-local.ipynb`](01-local.ipynb): open a large local read-only dataset by mounting it into the online lab
- [`02-remote.ipynb`](02-remote.ipynb): open large remote datasets using `fsspec`, `kerchunk`, and `zarr`
- [`03-cdsapi.ipynb`](03-cdsapi.ipynb): download small datasets from the Climate Data Store using the `cdsapi`
- [`04-ecmwfapi.ipynb`](04-ecmwfapi.ipynb): download small datasets from the ECMWF Archive using the `ecmwfapi`
- [`05-hplp-s3bucket.ipynb`](05-hplp-s3bucket.ipynb): open large hplp-experiment datasets from the ECMWF S3 bucket


## License

Licensed under the CC BY 4.0 license ([LICENSE](../LICENSE.txt) or https://creativecommons.org/licenses/by/4.0/).
