"""
Hierarchical-interpolation context-mixing codec (`InterpCtxCodec`).

Like SZ3's interpolation mode, each 2D slice is coded coarse-to-fine: the
coarsest sub-grid is predicted causally (MED), then every refinement level
predicts the new points by cubic (or linear, chosen per pass) interpolation
along one axis from already reconstructed points on both sides.  Prediction
residuals are quantised with step 2*eb (in-loop, so |x - x_dec| <= eb) and
coded with a context-mixing binary arithmetic coder using an Elias-gamma
binarisation (zero flag, sign, unary magnitude bucket, mantissa bits).
Contexts combine level/axis, the local stencil gradient and curvature, and the
magnitudes of neighbouring residuals (same level, and the previous slice).
"""

import struct

import numpy as np
from numba import njit
from numcodecs.abc import Codec
from numcodecs.registry import register_codec

from ctxcoder import (
    MASK32,
    MODEL_ONE,
    _dec_bit,
    _dec_init,
    _enc_bit,
    _enc_flush,
    _hash,
    _make_dt,
    _predict,
    _to_p12,
    _update,
)
from ctxcodec2 import _bucket, _med, _sbucket

NM = 7
TB = 22


@njit(cache=True)
def _code_bit(ctxs, node, wsel, bit, probs, counts, weights, idx, st, state, out, inp, encode, lr, dt, tmask):
    # ctxs: int64[8] context values; node: decision id
    idx[0] = _hash(node, ctxs[0], 0, 0, 1, tmask)
    idx[1] = _hash(node, ctxs[1], ctxs[2], 0, 2, tmask)
    idx[2] = _hash(node, ctxs[0], ctxs[3], 0, 3, tmask)
    idx[3] = _hash(node, ctxs[2], ctxs[4], ctxs[5], 4, tmask)
    idx[4] = _hash(node, ctxs[1], ctxs[5], ctxs[6], 5, tmask)
    idx[5] = _hash(node, ctxs[0], ctxs[7], ctxs[8], 6, tmask)
    idx[6] = _hash(node, ctxs[8], ctxs[9], ctxs[1], 7, tmask)
    pmix = _predict(probs, counts, idx, weights, wsel, st, NM)
    p12 = _to_p12(pmix)
    if encode:
        _enc_bit(state, out, p12, bit)
    else:
        bit = _dec_bit(state, inp, p12)
    _update(probs, counts, idx, weights, wsel, st, NM, pmix, bit, lr, dt)
    return bit


@njit(cache=True)
def _code_residual(r, ctxs, base, probs, counts, weights, idx, st, state, out, inp, encode, lr, dt, tmask):
    """Elias-gamma style binarisation of integer residual r; returns r."""
    ab = ctxs[1]
    wbase = base * 64
    # zero flag
    z = 1 if (encode and r == 0) else 0
    z = _code_bit(ctxs, wbase + 0, (wbase + 0) * 10 + ab, z, probs, counts, weights, idx, st, state, out, inp, encode, lr, dt, tmask)
    if z == 1:
        return 0
    # sign
    sgn = 1 if (encode and r < 0) else 0
    sgn = _code_bit(ctxs, wbase + 1, (wbase + 1) * 10 + ab, sgn, probs, counts, weights, idx, st, state, out, inp, encode, lr, dt, tmask)
    m = (abs(r) - 1) if encode else 0
    # unary bucket k = bit_length(m + 1) - 1
    if encode:
        k = 0
        t = m + 1
        while t > 1:
            t >>= 1
            k += 1
    else:
        k = 0
    kk = 0
    while True:
        if kk >= 30:
            break
        more = 1 if (encode and kk < k) else 0
        more = _code_bit(ctxs, wbase + 2 + kk, (wbase + 2 + min(kk, 12)) * 10 + ab, more, probs, counts, weights, idx, st, state, out, inp, encode, lr, dt, tmask)
        if more == 0:
            break
        kk += 1
    k = kk
    # mantissa bits (MSB first): m + 1 in [2^k, 2^(k+1))
    v = 1
    for b in range(k - 1, -1, -1):
        bit = ((m + 1) >> b) & 1 if encode else 0
        node = wbase + 40 + min(k, 15)
        bit = _code_bit(ctxs, node * 16 + min(b, 15), (wbase + 40) * 10 + ab, bit, probs, counts, weights, idx, st, state, out, inp, encode, lr, dt, tmask)
        v = v * 2 + bit
    m = v - 1
    r_dec = -(m + 1) if sgn == 1 else (m + 1)
    return r_dec


@njit(cache=True)
def _predict_interp(rec, i, j, h, along_x, Y, X, cubic):
    """Interpolate at (i, j) from grid points at distance h along an axis."""
    if along_x:
        have0 = j - 3 * h >= 0
        have3 = j + 3 * h < X
        have2 = j + h < X
        p1 = rec[i, j - h]
        p2 = rec[i, j + h] if have2 else 0.0
        p0 = rec[i, j - 3 * h] if have0 else 0.0
        p3 = rec[i, j + 3 * h] if have3 else 0.0
    else:
        have0 = i - 3 * h >= 0
        have3 = i + 3 * h < Y
        have2 = i + h < Y
        p1 = rec[i - h, j]
        p2 = rec[i + h, j] if have2 else 0.0
        p0 = rec[i - 3 * h, j] if have0 else 0.0
        p3 = rec[i + 3 * h, j] if have3 else 0.0
    if not have2:
        if have0:
            return 2.0 * p1 - p0, abs(p1 - p0), 0.0
        return p1, 0.0, 0.0
    grad = abs(p2 - p1)
    if not cubic:
        return 0.5 * (p1 + p2), grad, 0.0
    if have0 and have3:
        return (-p0 + 9.0 * p1 + 9.0 * p2 - p3) / 16.0, grad, abs(p1 + p2 - p0 - p3)
    if have0:
        return (-p0 + 6.0 * p1 + 3.0 * p2) / 8.0, grad, abs(2.0 * p1 - p0 - p2)
    if have3:
        return (3.0 * p1 + 6.0 * p2 - p3) / 8.0, grad, abs(2.0 * p2 - p1 - p3)
    return 0.5 * (p1 + p2), grad, 0.0


@njit(cache=True)
def _code_slice(x, rec, res, res_prev, has_prev, w, probs, counts, weights, idx, st, state, out, inp, encode, lr, dt, tmask, levels):
    Y, X = x.shape
    ctxs = np.zeros(10, np.int64)
    S = 1 << levels
    # ---- coarsest grid: causal MED prediction
    for i in range(0, Y, S):
        for j in range(0, X, S):
            L = rec[i, j - S] if j >= S else np.nan
            U = rec[i - S, j] if i >= S else np.nan
            UL = rec[i - S, j - S] if (i >= S and j >= S) else np.nan
            if j >= S and i >= S:
                pred = _med(L, U, UL)
                grad = abs(L - U)
            elif j >= S:
                pred = L
                grad = abs(L - rec[i, j - 2 * S]) if j >= 2 * S else 0.0
            elif i >= S:
                pred = U
                grad = abs(U - rec[i - 2 * S, j]) if i >= 2 * S else 0.0
            else:
                pred = 0.0
                grad = 0.0
            ab = _bucket(int(grad / w)) if grad / w < 1e9 else 9
            rL = (_sbucket(res[i, j - S]) + 11) if j >= S else 21
            rU = (_sbucket(res[i - S, j]) + 11) if i >= S else 21
            rP = (_sbucket(res_prev[i, j]) + 11) if has_prev else 21
            ctxs[0] = levels * 4 + 3
            ctxs[1] = ab
            ctxs[2] = rL
            ctxs[3] = 0
            ctxs[4] = rU
            ctxs[5] = rP
            ctxs[6] = 0
            ctxs[7] = levels
            ctxs[8] = (max(-15, min(15, res[i, j - S])) + 16) if j >= S else 0
            ctxs[9] = (max(-15, min(15, res[i - S, j])) + 16) if i >= S else 0
            r = int(np.rint((x[i, j] - pred) / w)) if encode else 0
            r = _code_residual(r, ctxs, 0, probs, counts, weights, idx, st, state, out, inp, encode, lr, dt, tmask)
            rec[i, j] = pred + r * w
            res[i, j] = r
    # ---- refinement levels
    for lev in range(levels, 0, -1):
        s = 1 << lev
        h = s >> 1
        for pas in range(2):
            along_x = pas == 0
            # choose cubic vs linear (encoder decides, 1 flag bit)
            cubic = 1
            if encode:
                e_c = 0.0
                e_l = 0.0
                if along_x:
                    for i in range(0, Y, s):
                        for j in range(h, X, s):
                            pc, g, c = _predict_interp(rec, i, j, h, True, Y, X, True)
                            pl, g, c = _predict_interp(rec, i, j, h, True, Y, X, False)
                            e_c += abs(x[i, j] - pc)
                            e_l += abs(x[i, j] - pl)
                else:
                    for i in range(h, Y, s):
                        for j in range(0, X, h):
                            pc, g, c = _predict_interp(rec, i, j, h, False, Y, X, True)
                            pl, g, c = _predict_interp(rec, i, j, h, False, Y, X, False)
                            e_c += abs(x[i, j] - pc)
                            e_l += abs(x[i, j] - pl)
                cubic = 1 if e_c <= e_l else 0
                _enc_bit(state, out, 2048, cubic)
            else:
                cubic = _dec_bit(state, inp, 2048)
            use_cubic = cubic == 1
            base = 1 + (2 if along_x else 1)
            if along_x:
                i0, istep, j0, jstep = 0, s, h, s
            else:
                i0, istep, j0, jstep = h, s, 0, h
            for i in range(i0, Y, istep):
                for j in range(j0, X, jstep):
                    pred, grad, curv = _predict_interp(rec, i, j, h, along_x, Y, X, use_cubic)
                    g = grad / w
                    ab = _bucket(int(g)) if g < 1e9 else 9
                    cv = curv / w
                    cb = _bucket(int(cv)) if cv < 1e9 else 9
                    # neighbouring residuals coded at this level/pass
                    if along_x:
                        rL = (_sbucket(res[i, j - s]) + 11) if j >= s else 21
                        rU = (_sbucket(res[i - s, j]) + 11) if i >= s else 21
                        rD = (_sbucket(res[i - h, j - h]) + 11) if (i >= h and j >= h) else 21
                    else:
                        rL = (_sbucket(res[i, j - h]) + 11) if j >= h else 21
                        rU = (_sbucket(res[i - s, j]) + 11) if i >= s else 21
                        rD = (_sbucket(res[i - h, j]) + 11) if i >= h else 21
                    rP = (_sbucket(res_prev[i, j]) + 11) if has_prev else 21
                    pb = _bucket(int(abs(pred) / w)) if abs(pred) / w < 1e9 else 9
                    ctxs[0] = lev * 4 + pas
                    ctxs[1] = ab
                    ctxs[2] = rL
                    ctxs[3] = cb
                    ctxs[4] = rU
                    ctxs[5] = rP
                    ctxs[6] = rD
                    ctxs[7] = pb
                    if along_x:
                        eL = res[i, j - s] if j >= s else 0
                        eU = res[i - s, j] if i >= s else 0
                    else:
                        eL = res[i, j - h] if j >= h else 0
                        eU = res[i - h, j] if i >= h else 0
                    ctxs[8] = max(-15, min(15, eL)) + 16
                    ctxs[9] = max(-15, min(15, eU)) + 16
                    r = int(np.rint((x[i, j] - pred) / w)) if encode else 0
                    r = _code_residual(r, ctxs, base, probs, counts, weights, idx, st, state, out, inp, encode, lr, dt, tmask)
                    rec[i, j] = pred + r * w
                    res[i, j] = r


@njit(cache=True)
def _code_all(x, rec, T, Y, X, w, levels, out, inp, encode, lr, lim):
    tsize = 1 << TB
    tmask = tsize - 1
    probs = np.full((NM, tsize), MODEL_ONE // 2, np.int32)
    counts = np.zeros((NM, tsize), np.uint8)
    weights = np.full((4096, NM + 1), 0.25, np.float64)
    idx = np.zeros(NM, np.int64)
    st = np.zeros(NM + 1, np.float64)
    dt = _make_dt(lim)
    state = np.zeros(5, np.int64)
    if encode:
        state[1] = MASK32
        state[3] = 1
    else:
        _dec_init(inp, state)
    res = np.zeros((Y, X), np.int32)
    res_prev = np.zeros((Y, X), np.int32)
    for t in range(T):
        _code_slice(x[t], rec[t], res, res_prev, t > 0, w, probs, counts, weights, idx, st, state, out, inp, encode, lr, dt, tmask, levels)
        for i in range(Y):
            for j in range(X):
                res_prev[i, j] = res[i, j]
    if encode:
        _enc_flush(state, out)
    return state[4]


class InterpCtxCodec(Codec):
    codec_id = "interp-ctx"

    def __init__(self, eb, lr=0.008, lim=1 / 256, levels=None, shrink=1e-6):
        self.eb = float(eb)
        self.lr = float(lr)
        self.lim = float(lim)
        self.levels = levels
        self.shrink = float(shrink)

    def _levels(self, Y, X):
        if self.levels is not None:
            return int(self.levels)
        return max(1, int(np.floor(np.log2(max(Y, X)))) - 2)

    def encode(self, buf):
        x = np.ascontiguousarray(buf)
        dtype, shape = x.dtype, x.shape
        x3 = np.ascontiguousarray(x.reshape((-1,) + shape[-2:]).astype(np.float64))
        T, Y, X = x3.shape
        w = 2.0 * self.eb * (1.0 - self.shrink)
        levels = self._levels(Y, X)
        rec = np.zeros_like(x3)
        out = np.zeros(x3.size * 8 + 4096, np.uint8)
        n = _code_all(x3, rec, T, Y, X, w, levels, out, np.zeros(1, np.uint8), True, self.lr, self.lim)
        header = struct.pack("<4sBBdddd", b"ICX1", len(shape), levels, w, self.lr, self.lim, self.eb)
        header += struct.pack("<%dq" % len(shape), *shape)
        header += dtype.str.encode().ljust(8, b"\0")
        return header + out[:n].tobytes()

    def decode(self, buf, out=None):
        buf = memoryview(buf).tobytes()
        hs = struct.calcsize("<4sBBdddd")
        magic, ndim, levels, w, lr, lim, eb = struct.unpack("<4sBBdddd", buf[:hs])
        assert magic == b"ICX1"
        off = hs
        shape = struct.unpack("<%dq" % ndim, buf[off:off + 8 * ndim])
        off += 8 * ndim
        dtype = np.dtype(buf[off:off + 8].rstrip(b"\0").decode())
        off += 8
        inp = np.concatenate([np.frombuffer(buf[off:], np.uint8), np.zeros(16, np.uint8)])
        T = int(np.prod(shape[:-2])) if ndim > 2 else 1
        Y, X = shape[-2], shape[-1]
        rec = np.zeros((T, Y, X), np.float64)
        _code_all(rec, rec, T, Y, X, w, levels, np.zeros(1, np.uint8), inp, False, lr, lim)
        if np.issubdtype(dtype, np.integer):
            rec = np.rint(rec)
        result = rec.reshape(shape).astype(dtype)
        if out is not None:
            out[...] = result
            return out
        return result


register_codec(InterpCtxCodec)
