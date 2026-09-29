"""Wrapper codecs used by the ERA5 challenge submissions."""

import struct

import numpy as np
from numba import njit
from numcodecs.abc import Codec
from numcodecs.registry import get_codec, register_codec

from ctxcoder import MASK32, _code_mask, _dec_init, _enc_flush

# make sure the context-mixing codecs are registered
import ctxcoder  # noqa: F401
import ctxcodec2  # noqa: F401


# ----------------------------------------------------------------------------
# standalone mask coding
# ----------------------------------------------------------------------------
@njit(cache=True)
def _enc_mask_only(m, T, Y, X, out):
    state = np.zeros(5, np.int64)
    state[1] = MASK32
    state[3] = 1
    dummy = np.zeros(1, np.uint8)
    _code_mask(m, T, Y, X, out, dummy, state, True, 0.01, 1.0 / 512.0)
    _enc_flush(state, out)
    return state[4]


@njit(cache=True)
def _dec_mask_only(inp, T, Y, X):
    state = np.zeros(5, np.int64)
    _dec_init(inp, state)
    m = np.zeros((T, Y, X), np.uint8)
    dummy = np.zeros(1, np.uint8)
    _code_mask(m, T, Y, X, dummy, inp, state, False, 0.01, 1.0 / 512.0)
    return m


def encode_mask(mask):
    m = np.ascontiguousarray(mask.astype(np.uint8))
    m3 = m.reshape((-1,) + m.shape[-2:]) if m.ndim >= 2 else m.reshape(1, 1, -1)
    T, Y, X = m3.shape
    out = np.zeros(m3.size // 4 + 1024, np.uint8)
    n = _enc_mask_only(m3, T, Y, X, out)
    return out[:n].tobytes()


def decode_mask(b, shape):
    shape3 = (int(np.prod(shape[:-2])), shape[-2], shape[-1]) if len(shape) >= 2 else (1, 1, shape[0])
    inp = np.concatenate([np.frombuffer(b, np.uint8), np.zeros(16, np.uint8)])
    return _dec_mask_only(inp, *shape3).reshape(shape).astype(bool)


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


def _nearest_fill(x, mask):
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


class MaskFillCodec(Codec):
    """
    Preserve NaNs and/or exact zeros via separately coded masks; masked
    positions are filled (nearest neighbour) before the inner codec.
    """

    codec_id = "mask-fill"

    def __init__(self, inner, mask_nan=True, mask_zero=False, fill="nearest"):
        self.inner = _cfg(inner)
        self.mask_nan = bool(mask_nan)
        self.mask_zero = bool(mask_zero)
        self.fill = fill

    def encode(self, buf):
        x = np.ascontiguousarray(buf)
        xf = x.astype(np.float64)
        nanm = np.isnan(xf) if self.mask_nan else np.zeros(x.shape, bool)
        zm = (xf == 0) & ~nanm if self.mask_zero else np.zeros(x.shape, bool)
        both = nanm | zm
        if both.any():
            if self.fill == "nearest":
                xf = _nearest_fill(xf, both)
            else:
                xf = np.where(both, 0.0, xf)
        inner_bytes = get_codec(self.inner).encode(xf.astype(x.dtype))
        nb = encode_mask(nanm) if nanm.any() else b""
        zb = encode_mask(zm) if zm.any() else b""
        header = b"MSK1" + _pack_shape_dtype(x.shape, x.dtype) + struct.pack("<qq", len(nb), len(zb))
        return header + nb + zb + bytes(inner_bytes)

    def decode(self, buf, out=None):
        buf = memoryview(buf).tobytes()
        assert buf[:4] == b"MSK1"
        shape, dtype, off = _unpack_shape_dtype(buf, 4)
        ln, lz = struct.unpack("<qq", buf[off:off + 16])
        off += 16
        nb = buf[off:off + ln]
        off += ln
        zb = buf[off:off + lz]
        off += lz
        x = np.asarray(_dec(self.inner, buf[off:])).reshape(shape).astype(dtype)
        if lz:
            x[decode_mask(zb, shape)] = 0
        if ln:
            x[decode_mask(nb, shape)] = np.nan
        if out is not None:
            out[...] = x
            return out
        return x


class ClipCodec(Codec):
    """Clip decoded values into [minimum, maximum]."""

    codec_id = "clip"

    def __init__(self, inner, minimum=None, maximum=None):
        self.inner = _cfg(inner)
        self.minimum = minimum
        self.maximum = maximum

    def encode(self, buf):
        x = np.ascontiguousarray(buf)
        return b"CLP1" + _pack_shape_dtype(x.shape, x.dtype) + bytes(get_codec(self.inner).encode(x))

    def decode(self, buf, out=None):
        buf = memoryview(buf).tobytes()
        assert buf[:4] == b"CLP1"
        shape, dtype, off = _unpack_shape_dtype(buf, 4)
        x = np.asarray(_dec(self.inner, buf[off:])).reshape(shape).astype(dtype)
        lo = -np.inf if self.minimum is None else self.minimum
        hi = np.inf if self.maximum is None else self.maximum
        np.clip(x, lo, hi, out=x)
        if out is not None:
            out[...] = x
            return out
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


class GridIntCodec(Codec):
    """
    Lossless codec for data lying exactly on a uniform grid offset + k*scale
    (e.g. GRIB-packed data).  The integers k are coded with the inner codec
    (as float64 with absolute error bound < 0.5); the reconstruction
    offset + k*scale must be bitwise exact, which the encoder verifies.
    """

    codec_id = "grid-int"

    def __init__(self, inner):
        self.inner = _cfg(inner)

    @staticmethod
    def detect(x):
        fin = x[np.isfinite(x)]
        u = np.unique(fin)
        if len(u) < 2:
            return float(u[0]) if len(u) else 0.0, 1.0
        d = np.diff(u)
        scale = float(d.min())
        # refine: gcd-like check
        ratios = d / scale
        if not np.allclose(ratios, np.rint(ratios), rtol=0, atol=1e-6):
            raise ValueError("data not on a uniform grid")
        return float(u[0]), scale

    def encode(self, buf):
        x = np.ascontiguousarray(buf)
        xf = x.astype(np.float64)
        offset, scale = self.detect(xf)
        k = np.rint((xf - offset) / scale)
        rec = (offset + k * scale).astype(x.dtype)
        fin = np.isfinite(xf)
        if not np.array_equal(rec[fin].view(np.uint64) if x.dtype == np.float64 else rec[fin], x[fin].view(np.uint64) if x.dtype == np.float64 else x[fin]):
            raise ValueError("grid reconstruction is not bitwise exact")
        k = np.where(fin, k, np.nan)
        header = b"GRD1" + _pack_shape_dtype(x.shape, x.dtype) + struct.pack("<dd", offset, scale)
        return header + bytes(get_codec(self.inner).encode(k))

    def decode(self, buf, out=None):
        buf = memoryview(buf).tobytes()
        assert buf[:4] == b"GRD1"
        shape, dtype, off = _unpack_shape_dtype(buf, 4)
        offset, scale = struct.unpack("<dd", buf[off:off + 16])
        off += 16
        k = np.asarray(_dec(self.inner, buf[off:]), dtype=np.float64).reshape(shape)
        x = (offset + np.rint(k) * scale).astype(dtype)
        x[np.isnan(k)] = np.nan
        if out is not None:
            out[...] = x
            return out
        return x


class PerSliceCodec(Codec):
    """Encode each leading-axis slice independently with the inner codec."""

    codec_id = "per-slice"

    def __init__(self, inner):
        self.inner = _cfg(inner)

    def encode(self, buf):
        x = np.ascontiguousarray(buf)
        parts = [bytes(get_codec(self.inner).encode(x[i])) for i in range(x.shape[0])]
        header = b"SLC1" + _pack_shape_dtype(x.shape, x.dtype) + struct.pack("<%dq" % len(parts), *[len(p) for p in parts])
        return header + b"".join(parts)

    def decode(self, buf, out=None):
        buf = memoryview(buf).tobytes()
        assert buf[:4] == b"SLC1"
        shape, dtype, off = _unpack_shape_dtype(buf, 4)
        n = shape[0]
        lens = struct.unpack("<%dq" % n, buf[off:off + 8 * n])
        off += 8 * n
        x = np.empty(shape, dtype)
        for i, ln in enumerate(lens):
            x[i] = np.asarray(_dec(self.inner, buf[off:off + ln])).reshape(shape[1:]).astype(dtype)
            off += ln
        if out is not None:
            out[...] = x
            return out
        return x


class PostLosslessCodec(Codec):
    """Apply a lossless byte codec (e.g. LZMA) to the inner codec's output."""

    codec_id = "post-lossless"

    def __init__(self, inner, lossless):
        self.inner = _cfg(inner)
        self.lossless = _cfg(lossless)

    def encode(self, buf):
        x = np.ascontiguousarray(buf)
        e = bytes(get_codec(self.inner).encode(x))
        return b"PLL1" + _pack_shape_dtype(x.shape, x.dtype) + bytes(get_codec(self.lossless).encode(np.frombuffer(e, np.uint8)))

    def decode(self, buf, out=None):
        buf = memoryview(buf).tobytes()
        assert buf[:4] == b"PLL1"
        shape, dtype, off = _unpack_shape_dtype(buf, 4)
        e = bytes(_dec(self.lossless, buf[off:]))
        x = np.asarray(_dec(self.inner, e)).reshape(shape).astype(dtype)
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
        nb = encode_mask(nanm) if nanm.any() else b""
        return b"CST1" + _pack_shape_dtype(x.shape, x.dtype) + struct.pack("<q", len(nb)) + nb

    def decode(self, buf, out=None):
        buf = memoryview(buf).tobytes()
        assert buf[:4] == b"CST1"
        shape, dtype, off = _unpack_shape_dtype(buf, 4)
        (ln,) = struct.unpack("<q", buf[off:off + 8])
        off += 8
        x = np.full(shape, self.value, dtype)
        if ln:
            x[decode_mask(buf[off:off + ln], shape)] = np.nan
        if out is not None:
            out[...] = x
            return out
        return x


for _c in (MaskFillCodec, ClipCodec, AbsRelCodec, GridIntCodec, PerSliceCodec, PostLosslessCodec, ConstantCodec, ThresholdCodec):
    register_codec(_c)
