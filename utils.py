import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Union
from urllib.parse import unquote as urlunquote
from urllib.parse import urlparse

import fcbench
import ipyfilite
import numcodecs
import pandas as pd
import xarray as xr
import zarr
from numcodecs.abc import Codec


def open_dataset(path: Path, **kwargs) -> xr.Dataset:
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
    uploader = ipyfilite.FileUploadLite()
    await uploader.request()
    uploader.close()

    return uploader.value[0].path


def mount_http_file(url: str, name: Optional[str] = None) -> Path:
    if name is None:
        name = _get_name_from_url(url)

    http_file = ipyfilite.HTTPFileIO(name=name, url=url)

    return http_file.path


def _get_name_from_url(url: str) -> str:
    return urlunquote(Path(urlparse(url).path).name)


async def download_dataset_as_zarr(
    ds: xr.Dataset,
    name: str,
    compressor: Union[Codec, list[Codec], dict[str, Union[Codec, list[Codec]]]],
    zip_compression: int = 0,
):
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
    try:
        async with ipyfilite.FileDownloadPathLite(name) as path:
            yield path
    finally:
        pass


def format_compress_stats(
    codecs: list[numcodecs.abc.Codec],
    stats: list[fcbench.compressor.types.CodecPerformanceMeasurement],
):
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
            round(stat.encode_instructions / stat.decoded_bytes, 1),
            round(stat.decode_instructions / stat.decoded_bytes, 1),
        ]

    return table


# TODO: move this into the lab-specific patches
try:
    import js
    import pyodide_http
    import urllib
except ImportError:
    pass
else:
    _old_send = pyodide_http._core.send

    def _new_send(request: pyodide_http._core.Request, stream: bool = False) -> pyodide_http._core.Response:
        if js.URL.new(request.url).origin != js.location.origin:
            request.url = "https://proxy.climet.eu/" + request.url
        return _old_send(request, stream)

    pyodide_http._core.send = _new_send

    def _new_urlopen(url, *args, **kwargs):
        from io import BytesIO

        import urllib.request
        from http.client import HTTPResponse

        from pyodide_http._core import Request, send
        from pyodide_http._urllib import FakeSock

        method = "GET"
        data = None
        headers = {}
        if isinstance(url, urllib.request.Request):
            method = url.get_method()
            data = url.data
            headers = dict(url.header_items())
            url = url.full_url

        request = Request(method, url, headers=headers, body=data)
        resp = send(request)

        # Build a fake http response
        # Strip out the content-length header. When Content-Encoding is 'gzip' (or other
        # compressed format) the 'Content-Length' is the compressed length, while the
        # data itself is uncompressed. This will cause problems while decoding our
        # fake response.
        headers_without_content_length = {
            k: v for k, v in resp.headers.items() if k != "content-length"
        } if "content-encoding" in resp.headers.keys() else resp.headers
        response_data = (
            b"HTTP/1.1 "
            + str(resp.status_code).encode("ascii")
            + b"\n"
            + "\n".join(
                f"{'_'.join(k.title() for k in key.split('_'))}: {value}" for key, value in headers_without_content_length.items()
            ).encode("ascii")
            + b"\n\n"
            + resp.body
        )

        response = HTTPResponse(FakeSock(response_data))
        response.begin()
        return response

    def _new_urlopen_self_removed(self, url, *args, **kwargs):
        return new_urlopen(url, *args, **kwargs)

    urllib.request.urlopen = _new_urlopen
    urllib.request.OpenerDirector.open = _new_urlopen_self_removed
