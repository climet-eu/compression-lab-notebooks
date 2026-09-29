"""
Codec for an absolute error bound on the (periodic) longitude derivative

    dX/dlon[i] = (x[i+k] - x[i-k]) / (2*k*dl)

Only the stride-2k differences D[i] = x[i+k] - x[i-k] are constrained
(|D_dec - D| <= eb_qoi * 2*k*dl), so we compress D directly with an absolute
error bound and reconstruct x by integrating D along each residue class
(mod 2k) of the periodic longitude axis.  The per-class integration constant
is free w.r.t. the requirement; we store the class means coarsely so that the
reconstruction stays close to the original field.

The periodic closure (sum of D over a class must be zero) is enforced by a
ramp correction on the decoder side; the encoder shrinks the quantisation
step until the effective error bound holds everywhere.
"""

import lzma
import struct

import numpy as np
from numba import njit
from numcodecs.abc import Codec
from numcodecs.registry import get_codec, register_codec

from ctxcodec2 import CtxCodec  # noqa: F401  (registers "ctx-mixing")


def _reconstruct(D, anchors, k, a_step):
    """D: (T, Y, X) decoded differences, anchors: (T, Y, 2k) int class means."""
    T, Y, X = D.shape
    n = 2 * k
    K = X // n
    Droll = np.roll(D, -k, axis=2).reshape(T, Y, K, n)
    S = Droll.sum(axis=2, keepdims=True)
    Deff = Droll - S / K
    z = np.zeros((T, Y, K, n), np.float64)
    z[:, :, 1:, :] = np.cumsum(Deff[:, :, :-1, :], axis=2)
    z -= z.mean(axis=2, keepdims=True)
    z += (anchors.astype(np.float64) * a_step)[:, :, None, :]
    return z.reshape(T, Y, X)


@njit(cache=True)
def _optimise_classes(e_r, q_r, w, T, Y, K, n, max_flips):
    """For every class pick the number m of rounding flips (of the m largest
    same-sign errors) that minimises the final max error after the closure
    ramp, max_k |e'_k - mean(e')|.  Modifies e_r and q_r in place."""
    vals = np.empty(K, np.float64)
    idxs = np.empty(K, np.int64)
    flipped = np.zeros(K, np.bool_)
    for t in range(T):
        for y in range(Y):
            for c in range(n):
                S = 0.0
                for k in range(K):
                    S += e_r[t, y, k, c]
                sgn = 1.0 if S > 0 else -1.0
                cnt = 0
                for k in range(K):
                    ek = e_r[t, y, k, c]
                    if (ek > 0 and sgn > 0) or (ek < 0 and sgn < 0):
                        vals[cnt] = -abs(ek)
                        idxs[cnt] = k
                        cnt += 1
                order = np.argsort(vals[:cnt])
                best_m = 0
                # error for m = 0
                mean = S / K
                best_err = 0.0
                for k in range(K):
                    v = abs(e_r[t, y, k, c] - mean)
                    if v > best_err:
                        best_err = v
                mmax = min(cnt, max_flips)
                for k in range(K):
                    flipped[k] = False
                for m in range(1, mmax + 1):
                    flipped[idxs[order[m - 1]]] = True
                    Sp = S - m * w * sgn
                    mean = Sp / K
                    err = 0.0
                    for k in range(K):
                        ek = e_r[t, y, k, c]
                        if flipped[k]:
                            ek -= sgn * w
                        v = abs(ek - mean)
                        if v > err:
                            err = v
                    if err < best_err:
                        best_err = err
                        best_m = m
                for m in range(best_m):
                    k = idxs[order[m]]
                    e_r[t, y, k, c] -= sgn * w
                    q_r[t, y, k, c] -= sgn


def _closure_aware_quantise(D, w, k, ebD):
    """Round D onto the grid Dmin + w*q with per-class closure-aware flips."""
    T, Y, X = D.shape
    n = 2 * k
    K = X // n
    ymin = float(D.min())
    q = np.rint((D - ymin) / w)
    e = (ymin + w * q) - D
    e_r = np.ascontiguousarray(np.roll(e, -k, axis=2).reshape(T, Y, K, n))
    q_r = np.ascontiguousarray(np.roll(q, -k, axis=2).reshape(T, Y, K, n))
    _optimise_classes(e_r, q_r, w, T, Y, K, n, 48)
    q2 = np.roll(q_r.reshape(T, Y, X), k, axis=2)
    return ymin + w * q2


class LonGradientCodec(Codec):
    codec_id = "lon-gradient-difference"

    def __init__(self, eb_qoi, dl=0.25, offset=5, delta=0.04, anchor_step=None,
                 inner=None, lossless=None, flips=True):
        self.flips = bool(flips)
        self.eb_qoi = float(eb_qoi)
        self.dl = float(dl)
        self.offset = int(offset)
        self.delta = float(delta)
        self.anchor_step = anchor_step
        self.inner = inner if inner is not None else {"id": "ctx-mixing", "eb": 1.0, "mode": "abs", "shrink": 0.0}
        if isinstance(self.inner, Codec):
            self.inner = self.inner.get_config()
        self.lossless = lossless.get_config() if isinstance(lossless, Codec) else lossless
        self._last_info = None

    @property
    def eb_diff(self):
        return self.eb_qoi * 2 * self.offset * self.dl

    def _diff(self, x):
        k = self.offset
        return np.roll(x, -k, axis=-1) - np.roll(x, k, axis=-1)

    def encode(self, buf):
        x = np.ascontiguousarray(buf)
        dtype, shape = x.dtype, x.shape
        x3 = x.reshape((-1,) + shape[-2:]).astype(np.float64)
        T, Y, X = x3.shape
        k = self.offset
        n = 2 * k
        assert X % n == 0, "longitude length must be a multiple of 2*offset"
        K = X // n
        ebD = self.eb_diff
        a_step = self.anchor_step if self.anchor_step is not None else 0.8 * ebD
        D = self._diff(x3)
        # class means -> anchors
        M = x3.reshape(T, Y, K, n).mean(axis=2)
        anchors = np.rint(M / a_step).astype(np.int64)
        a_delta = np.diff(anchors, axis=2, prepend=0)  # along class
        a_delta[:, :, 0] = np.diff(anchors[:, :, 0], axis=1, prepend=0)
        a_bytes = lzma.compress(a_delta.astype(np.int32).tobytes(), preset=9 | lzma.PRESET_EXTREME)

        delta = self.delta
        while True:
            w = 2.0 * ebD * (1.0 - delta)
            cfg = dict(self.inner)
            cfg["eb"] = w / 2.0
            inner = get_codec(cfg)
            D_in = _closure_aware_quantise(D, w, k, ebD) if self.flips else D
            e_D = inner.encode(D_in)
            D_dec = inner.decode(e_D)
            x_dec = _reconstruct(D_dec, anchors, k, a_step)
            err = np.abs(self._diff(x_dec) - D).max()
            if err <= ebD * (1.0 - 1e-9):
                break
            delta += 0.02
            if delta > 0.5:
                raise RuntimeError("could not satisfy error bound")
        if self.lossless is not None:
            e_D = get_codec(self.lossless).encode(e_D)
        self._last_info = {"delta": delta, "w": w, "max_diff_err": float(err), "anchor_bytes": len(a_bytes), "diff_bytes": len(e_D)}
        header = struct.pack("<4sBddi", b"LGD1", len(shape), a_step, delta, len(a_bytes))
        header += struct.pack("<%dq" % len(shape), *shape)
        header += dtype.str.encode().ljust(8, b"\0")
        return header + a_bytes + bytes(e_D)

    def decode(self, buf, out=None):
        buf = memoryview(buf).tobytes()
        hs = struct.calcsize("<4sBddi")
        magic, ndim, a_step, delta, na = struct.unpack("<4sBddi", buf[:hs])
        assert magic == b"LGD1"
        off = hs
        shape = struct.unpack("<%dq" % ndim, buf[off:off + 8 * ndim])
        off += 8 * ndim
        dtype = np.dtype(buf[off:off + 8].rstrip(b"\0").decode())
        off += 8
        a_bytes = buf[off:off + na]
        off += na
        e_D = buf[off:]
        if self.lossless is not None:
            e_D = get_codec(self.lossless).decode(e_D)
        k = self.offset
        n = 2 * k
        T = int(np.prod(shape[:-2])) if ndim > 2 else 1
        Y, X = shape[-2], shape[-1]
        a_delta = np.frombuffer(lzma.decompress(a_bytes), np.int32).reshape(T, Y, n).astype(np.int64)
        anchors = a_delta.copy()
        anchors[:, :, 0] = np.cumsum(a_delta[:, :, 0], axis=1)
        anchors = np.cumsum(anchors, axis=2)
        w = 2.0 * self.eb_diff * (1.0 - delta)
        cfg = dict(self.inner)
        cfg["eb"] = w / 2.0
        D_dec = get_codec(cfg).decode(e_D)
        x_dec = _reconstruct(D_dec, anchors, k, a_step).reshape(shape).astype(dtype)
        if out is not None:
            out[...] = x_dec
            return out
        return x_dec


register_codec(LonGradientCodec)
