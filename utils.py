from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Union


def open_dataset(path: Path, **kwargs) -> "xarray.Dataset":
    import xarray as xr

    path = Path(path)

    if path.suffix == ".grib" or kwargs.get("engine", None) == "cfgrib":
        if "engine" not in kwargs:
            kwargs["engine"] = "cfgrib"
        if "backend_kwargs" not in kwargs:
            kwargs["backend_kwargs"] = dict()
        if "indexpath" not in kwargs["backend_kwargs"]:
            # cfgrib creates index files right next to the data file,
            #  which may be in a read-only file system
            kwargs["backend_kwargs"]["indexpath"] = ""

    if "".join(path.suffixes).endswith(".zarr.zip"):
        if "engine" not in kwargs:
            kwargs["engine"] = "zarr"

    if "chunks" not in kwargs:
        kwargs["chunks"] = "auto"

    if "cache" not in kwargs:
        kwargs["cache"] = False

    ds = xr.open_dataset(str(path), **kwargs)
    ds.attrs["path"] = str(path)

    return ds


async def mount_user_local_file() -> Path:
    import ipyfilite

    uploader = ipyfilite.FileUploadLite()
    await uploader.request()
    uploader.close()

    return uploader.value[0].path


def mount_http_file(url: str, name: Optional[str] = None) -> Path:
    import ipyfilite

    if name is None:
        name = _get_name_from_url(url)

    http_file = ipyfilite.HTTPFileIO(name=name, url=url)

    return http_file.path


def _get_name_from_url(url: str) -> str:
    from urllib.parse import unquote as urlunquote
    from urllib.parse import urlparse

    return urlunquote(Path(urlparse(url).path).name)


async def download_dataset_as_zarr(
    ds: "xarray.Dataset",
    name: str,
    compressor: Union[
        "numcodecs.abc.Codec",
        list["numcodecs.abc.Codec"],
        dict[str, Union["numcodecs.abc.Codec", list["numcodecs.abc.Codec"]]],
    ],
    zip_compression: int = 0,
):
    import ipyfilite
    import zarr

    name_suffix = "".join(Path(name).suffixes)

    # Ensure that the file path is easily recognisable as a zipped zarr file
    if name_suffix.endswith(".zarr.zip"):
        pass
    elif name_suffix.endswith(".zarr"):
        name = f"{name}.zip"
    elif name_suffix.endswith(".zip"):
        name = f"{Path(name).stem}.zarr.zip"
    else:
        name = f"{name}.zarr.zip"

    async with ipyfilite.FileDownloadPathLite(name) as path:
        store = zarr.storage.MemoryStore()
        chunk_store = zarr.storage.ZipStore(
            str(path),
            compression=zip_compression,
            allowZip64=True,
            mode="x",
        )

        compressors = (
            compressor
            if isinstance(compressor, dict)
            else {var: compressor for var in ds}
        )

        encoding = dict()
        for var, compressor in compressors.items():
            if isinstance(compressor, list):
                if len(compressor) == 0:
                    continue
                encoding[var] = dict(
                    compressor=compressor[0],
                    filters=compressor[1:],
                )
            else:
                encoding[var] = dict(
                    compressor=compressor,
                    filters=[],
                )

        ds.to_zarr(store=store, mode="w-", encoding=encoding)

        for key in store.keys():
            chunk_store[key] = store[key]

        store.close()
        chunk_store.close()


@asynccontextmanager
async def file_download_path(name: str) -> Path:
    import ipyfilite

    try:
        async with ipyfilite.FileDownloadPathLite(name) as path:
            yield path
    finally:
        pass


def format_compress_stats(
    codecs: list["numcodecs.abc.Codec"],
    stats: list["fcbench.compressor.types.CodecPerformanceMeasurement"],
):
    import pandas as pd

    table = pd.DataFrame(
        {
            "Codec": [],
            "compression ratio [raw B / enc B]": [],
            "encode throughput [raw GB/s]": [],
            "decode throughput [raw GB/s]": [],
            "encode instructions [#/B]": [],
            "decode instructions [#/B]": [],
        }
    ).set_index(["Codec"])

    for codec, stat in zip(codecs, stats):
        table.loc[str(codec), :] = [
            round(stat.decoded_bytes / stat.encoded_bytes, 2),
            round(
                1e-9
                * stat.decoded_bytes
                / (stat.encode_timing.secs + stat.encode_timing.nanos * 1e-9),
                2,
            ),
            round(
                1e-9
                * stat.decoded_bytes
                / (stat.decode_timing.secs + stat.decode_timing.nanos * 1e-9),
                2,
            ),
            round(stat.encode_instructions / stat.decoded_bytes, 1)
            if stat.encode_instructions is not None
            else None,
            round(stat.decode_instructions / stat.decoded_bytes, 1)
            if stat.decode_instructions is not None
            else None,
        ]

    return table


def kerchunk_autochunk(kc: dict, *, chunk_size: int) -> dict:
    import json

    import kerchunk
    import numpy as np
    import sympy

    kc_new = kc

    # iterate over all variables
    for k, v in kc["refs"].items():
        if Path(k).name == ".zarray":
            v = json.loads(v)

            # we cannot chunk compressed arrays
            if v["compressor"] is not None:
                continue

            chunks = v["chunks"]

            # calculate the size of the chunk
            nbytes_chunk = np.dtype(v["dtype"]).itemsize * int(
                np.prod(chunks, dtype=np.uint64)
            )

            # skip to the next variable if the chunk is already small enough
            if nbytes_chunk <= chunk_size:
                continue

            for i, c in enumerate(chunks):
                # factorize the remaining chunk size
                factors = []
                for f, c in sympy.factorint(c).items():
                    for _ in range(c):
                        factors.append(f)
                factors.sort()

                if len(factors) == 0:
                    continue

                # find the smallest factor that would reduce the chunk size
                #  below the limit, or the total factor
                factor = 1
                for f in factors:
                    factor *= f
                    if (nbytes_chunk // factor) <= chunk_size:
                        break

                # use kerchunk to apply the new chunking
                kc_new = kerchunk.utils.subchunk(
                    kc_new, Path(k).parts[0], factor,
                )

                chunks[i] = chunks[i] // factor
                nbytes_chunk = nbytes_chunk // factor

                # break if the chunk is now small enough
                if nbytes_chunk <= chunk_size:
                    break

    return kc_new
