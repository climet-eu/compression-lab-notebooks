"""Remaining wrapper codecs used by the ERA5 challenge submissions (to be
replaced by numcodecs-abs-or-rel, numcodecs-replace and numcodecs-zero).

Clipping, chunking, integer-grid coding, lossless post-compression, masking and
the entropy coders are provided by numcodecs-clip, numcodecs-chunked,
numcodecs-grid-int, numcodecs-combinators, numcodecs-mask, numcodecs-eb-quantize,
numcodecs-context-mixing and numcodecs-interp-ctx."""

import struct

import numpy as np
from numcodecs.abc import Codec
from numcodecs.registry import get_codec, register_codec




def _cfg(c):
    """Accept a codec config dict or a Codec instance."""
    return c.get_config() if isinstance(c, Codec) else c


def _dec(cfg, b):
    """Decode with an inner codec config; wasm codecs need a uint8 array."""
    return get_codec(cfg).decode(np.frombuffer(bytes(b), np.uint8))


def _pack_shape_dtype(shape, dtype):
    return struct.pack("<B%dq" % len(shape), len(shape), *shape) + np.dtype(dtype).str.encode().ljust(8, b"\0")


def _unpack_shape_dtype(buf, off):
    ndim = buf[off]
    off += 1
    shape = struct.unpack("<%dq" % ndim, buf[off:off + 8 * ndim])
    off += 8 * ndim
    dtype = np.dtype(buf[off:off + 8].rstrip(b"\0").decode())
    off += 8
    return shape, dtype, off


def _unused_nearest_fill(x, mask):
    """Fill masked values with the nearest unmasked value (per 2D slice)."""
    from scipy import ndimage

    x = x.copy()
    x3 = x.reshape((-1,) + x.shape[-2:])
    m3 = mask.reshape(x3.shape)
    for t in range(x3.shape[0]):
        if not m3[t].any():
            continue
        if m3[t].all():
            x3[t] = 0.0
            continue
        idx = ndimage.distance_transform_edt(m3[t], return_distances=False, return_indices=True)
        x3[t] = x3[t][tuple(idx)]
    return x


class AbsRelCodec(Codec):
    """
    Transform for a pointwise "abs OR rel" error bound.  y = f(x) with
    f'(x) = 1 / max(eb_abs, eb_rel*|x|); an inner codec with absolute error
    bound ln(1+eb_rel)/eb_rel on y guarantees |x_dec - x| <= max(eb_abs, eb_rel*|x|).
    """

    codec_id = "abs-or-rel-transform"

    def __init__(self, inner, eb_abs, eb_rel):
        self.inner = _cfg(inner)
        self.eb_abs = float(eb_abs)
        self.eb_rel = float(eb_rel)

    @property
    def eb_y(self):
        return np.log1p(self.eb_rel) / self.eb_rel

    def _fwd(self, x):
        a, r = self.eb_abs, self.eb_rel
        x0 = a / r
        ax = np.abs(x)
        y = np.where(ax <= x0, ax / a, x0 / a + np.log(np.maximum(ax, x0) / x0) / r)
        return np.sign(x) * y

    def _inv(self, y):
        a, r = self.eb_abs, self.eb_rel
        x0 = a / r
        y0 = x0 / a
        ay = np.abs(y)
        x = np.where(ay <= y0, ay * a, x0 * np.exp((np.maximum(ay, y0) - y0) * r))
        return np.sign(y) * x

    def encode(self, buf):
        x = np.ascontiguousarray(buf)
        y = self._fwd(x.astype(np.float64))
        return b"ABR1" + _pack_shape_dtype(x.shape, x.dtype) + bytes(get_codec(self.inner).encode(y))

    def decode(self, buf, out=None):
        buf = memoryview(buf).tobytes()
        assert buf[:4] == b"ABR1"
        shape, dtype, off = _unpack_shape_dtype(buf, 4)
        y = np.asarray(_dec(self.inner, buf[off:]), dtype=np.float64).reshape(shape)
        x = self._inv(y).astype(dtype)
        if out is not None:
            out[...] = x
            return out
        return x


class ThresholdCodec(Codec):
    """Set values with |x| < threshold to zero before the inner codec (uses
    part of a mean error budget to drop negligible values entirely)."""

    codec_id = "threshold-to-zero"

    def __init__(self, inner, threshold):
        self.inner = _cfg(inner)
        self.threshold = float(threshold)

    def encode(self, buf):
        x = np.ascontiguousarray(buf)
        xt = np.where(np.abs(x) < self.threshold, x.dtype.type(0), x)
        return b"THR1" + _pack_shape_dtype(x.shape, x.dtype) + bytes(get_codec(self.inner).encode(xt))

    def decode(self, buf, out=None):
        buf = memoryview(buf).tobytes()
        assert buf[:4] == b"THR1"
        shape, dtype, off = _unpack_shape_dtype(buf, 4)
        x = np.asarray(_dec(self.inner, buf[off:])).reshape(shape).astype(dtype)
        if out is not None:
            out[...] = x
            return out
        return x


class ConstantCodec(Codec):
    """Reconstruct the field as a single constant value (for trivially
    satisfiable requirements); NaNs are preserved via a mask."""

    codec_id = "constant-field"

    def __init__(self, value):
        self.value = float(value)

    def encode(self, buf):
        x = np.ascontiguousarray(buf)
        nanm = np.isnan(x.astype(np.float64))
        nb = bytes(get_codec({"id": "context_mixing.bitmap"}).encode(nanm)) if nanm.any() else b""
        return b"CST1" + _pack_shape_dtype(x.shape, x.dtype) + struct.pack("<q", len(nb)) + nb

    def decode(self, buf, out=None):
        buf = memoryview(buf).tobytes()
        assert buf[:4] == b"CST1"
        shape, dtype, off = _unpack_shape_dtype(buf, 4)
        (ln,) = struct.unpack("<q", buf[off:off + 8])
        off += 8
        x = np.full(shape, self.value, dtype)
        if ln:
            x[np.asarray(_dec({"id": "context_mixing.bitmap"}, buf[off:off + ln])).reshape(shape)] = np.nan
        if out is not None:
            out[...] = x
            return out
        return x


for _c in (AbsRelCodec, ConstantCodec, ThresholdCodec):
    register_codec(_c)
