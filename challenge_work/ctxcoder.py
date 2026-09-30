"""
Context-mixing binary arithmetic coder (numba-accelerated) and a codec for
error-bounded compression of gridded data with missing (NaN) values.

The codec `NanContextCodec`:
  * quantises finite values onto a uniform grid of width `2*eb` (so that the
    reconstruction error is <= eb), and
  * codes the NaN mask and the quantisation indices with a context-mixing
    arithmetic coder (PAQ-style logistic mixing of several context models),
    using 2D neighbours in the same slice and the previous slice as contexts.

Everything is deterministic and the decoder mirrors the encoder exactly.
"""

import math
import struct

import numpy as np
from numba import njit
from numcodecs.abc import Codec
from numcodecs.registry import register_codec

# ----------------------------------------------------------------------------
# LZMA-style binary range coder, 12-bit probabilities
# ----------------------------------------------------------------------------
PROB_BITS = 12
PROB_ONE = 1 << PROB_BITS
TOP = 1 << 24
MASK32 = 0xFFFFFFFF

# model probability precision (16 bits) and adaptation schedule
MODEL_ONE = 1 << 16
ADAPT_LIMIT = 255


@njit(cache=True)
def _shift_low(low, cache, cache_size, out, pos):
    if low < 0xFF000000 or low >= (1 << 32):
        carry = low >> 32
        temp = cache
        while True:
            out[pos] = (temp + carry) & 0xFF
            pos += 1
            temp = 0xFF
            cache_size -= 1
            if cache_size == 0:
                break
        cache = (low >> 24) & 0xFF
    cache_size += 1
    low = (low & 0x00FFFFFF) << 8
    return low, cache, cache_size, pos


@njit(cache=True)
def _enc_bit(state, out, p1, bit):
    """Encode `bit` with probability p1 = P(bit=1) in 12 bits (1..4095)."""
    low, rng, cache, cache_size, pos = state[0], state[1], state[2], state[3], state[4]
    p0 = PROB_ONE - p1
    bound = (rng >> PROB_BITS) * p0
    if bit == 0:
        rng = bound
    else:
        low += bound
        rng -= bound
    while rng < TOP:
        rng = (rng << 8) & MASK32
        low, cache, cache_size, pos = _shift_low(low, cache, cache_size, out, pos)
    state[0], state[1], state[2], state[3], state[4] = low, rng, cache, cache_size, pos


@njit(cache=True)
def _enc_flush(state, out):
    low, cache, cache_size, pos = state[0], state[2], state[3], state[4]
    for _ in range(5):
        low, cache, cache_size, pos = _shift_low(low, cache, cache_size, out, pos)
    state[0], state[2], state[3], state[4] = low, cache, cache_size, pos


@njit(cache=True)
def _dec_init(inp, state):
    code = 0
    pos = 0
    for _ in range(5):
        code = ((code << 8) | inp[pos]) & MASK32
        pos += 1
    state[0] = code
    state[1] = MASK32
    state[4] = pos


@njit(cache=True)
def _dec_bit(state, inp, p1):
    code, rng, pos = state[0], state[1], state[4]
    p0 = PROB_ONE - p1
    bound = (rng >> PROB_BITS) * p0
    if code < bound:
        rng = bound
        bit = 0
    else:
        code -= bound
        rng -= bound
        bit = 1
    while rng < TOP:
        rng = (rng << 8) & MASK32
        code = ((code << 8) | inp[pos]) & MASK32
        pos += 1
    state[0], state[1], state[4] = code, rng, pos
    return bit


# ----------------------------------------------------------------------------
# Context mixing
# ----------------------------------------------------------------------------
@njit(cache=True)
def _stretch(p):
    return math.log(p / (1.0 - p))


@njit(cache=True)
def _squash(x):
    if x > 30.0:
        x = 30.0
    elif x < -30.0:
        x = -30.0
    return 1.0 / (1.0 + math.exp(-x))


@njit(cache=True)
def _predict(probs, counts, idx, weights, wsel, st, nmodels):
    """Compute mixed P(bit=1) from model slots `idx[i]` and weight set `wsel`."""
    dot = 0.0
    for i in range(nmodels):
        p = (probs[i, idx[i]] + 0.5) / MODEL_ONE
        s = _stretch(p)
        st[i] = s
        dot += weights[wsel, i] * s
    # bias input
    st[nmodels] = 0.3
    dot += weights[wsel, nmodels] * 0.3
    return _squash(dot)


@njit(cache=True)
def _update(probs, counts, idx, weights, wsel, st, nmodels, pmix, bit, lr, dt):
    err = (bit - pmix) * lr
    for i in range(nmodels + 1):
        weights[wsel, i] += err * st[i]
    for i in range(nmodels):
        j = idx[i]
        n = counts[i, j]
        target = MODEL_ONE - 1 if bit == 1 else 0
        p = probs[i, j]
        p = p + int((target - p) * dt[n])
        if p < 1:
            p = 1
        elif p > MODEL_ONE - 1:
            p = MODEL_ONE - 1
        probs[i, j] = p
        if n < ADAPT_LIMIT:
            counts[i, j] = n + 1


@njit(cache=True)
def _to_p12(pmix):
    p = int(pmix * PROB_ONE)
    if p < 1:
        p = 1
    elif p > PROB_ONE - 1:
        p = PROB_ONE - 1
    return p


@njit(cache=True)
def _make_dt(limit_rate):
    dt = np.empty(ADAPT_LIMIT + 1, np.float64)
    for n in range(ADAPT_LIMIT + 1):
        r = 1.0 / (n + 1.5)
        if r < limit_rate:
            r = limit_rate
        dt[n] = r
    return dt


@njit(cache=True)
def _hash(a, b, c, d, e, mask):
    h = a * 0x9E3779B1 + b
    h = (h ^ (h >> 15)) * 0x85EBCA77 + c
    h = (h ^ (h >> 13)) * 0xC2B2AE3D + d
    h = (h ^ (h >> 16)) * 0x27D4EB2F + e
    h = h ^ (h >> 15)
    return h & mask


# ----------------------------------------------------------------------------
# Mask model: contexts from the current slice's causal neighbourhood and the
# previous slice.
# ----------------------------------------------------------------------------
MASK_NMODELS = 5
MASK_TABLE_BITS = 22


@njit(cache=True)
def _mask_ctx(m, t, i, j, T, Y, X, idx, mask):
    # neighbours in the current slice (causal)
    def g(tt, ii, jj):
        if ii < 0 or ii >= Y or jj < 0 or jj >= X or tt < 0:
            return 2  # outside
        return m[tt, ii, jj]

    L = g(t, i, j - 1)
    LL = g(t, i, j - 2)
    L3 = g(t, i, j - 3)
    U = g(t, i - 1, j)
    UL = g(t, i - 1, j - 1)
    UR = g(t, i - 1, j + 1)
    ULL = g(t, i - 1, j - 2)
    URR = g(t, i - 1, j + 2)
    UU = g(t, i - 2, j)
    UUL = g(t, i - 2, j - 1)
    UUR = g(t, i - 2, j + 1)
    U3 = g(t, i - 3, j)
    P = g(t - 1, i, j)
    PL = g(t - 1, i, j - 1)
    PR = g(t - 1, i, j + 1)
    PU = g(t - 1, i - 1, j)
    PD = g(t - 1, i + 1, j)

    c1 = ((((L * 3 + U) * 3 + UL) * 3 + UR) * 3 + LL) * 3 + UU
    c2 = ((((c1 * 3 + ULL) * 3 + URR) * 3 + UUL) * 3 + UUR) * 3 + L3
    c3 = (((P * 3 + PL) * 3 + PR) * 3 + PU) * 3 + PD
    idx[0] = _hash(c1, 1, 0, 0, 0, mask)
    idx[1] = _hash(c2, 2, U3, 0, 0, mask)
    idx[2] = _hash(c1, 3, c3, 0, 0, mask)
    idx[3] = _hash(c2, 4, c3, U3, 0, mask)
    # run-length like context: distance to last transition in row (capped)
    d = 0
    while j - 1 - d >= 0 and d < 32 and m[t, i, j - 1 - d] == L:
        d += 1
    idx[4] = _hash(L, 5, U, d, P, mask)
    return L * 3 + U


@njit(cache=True)
def _code_mask(m, T, Y, X, out, inp, state, encode, lr, lim):
    probs = np.full((MASK_NMODELS, 1 << MASK_TABLE_BITS), MODEL_ONE // 2, np.int32)
    counts = np.zeros((MASK_NMODELS, 1 << MASK_TABLE_BITS), np.uint8)
    weights = np.full((16, MASK_NMODELS + 1), 0.3, np.float64)
    idx = np.zeros(MASK_NMODELS, np.int64)
    st = np.zeros(MASK_NMODELS + 1, np.float64)
    dt = _make_dt(lim)
    tmask = (1 << MASK_TABLE_BITS) - 1
    for t in range(T):
        for i in range(Y):
            for j in range(X):
                wsel = _mask_ctx(m, t, i, j, T, Y, X, idx, tmask)
                pmix = _predict(probs, counts, idx, weights, wsel, st, MASK_NMODELS)
                p12 = _to_p12(pmix)
                if encode:
                    bit = m[t, i, j]
                    _enc_bit(state, out, p12, bit)
                else:
                    bit = _dec_bit(state, inp, p12)
                    m[t, i, j] = bit
                _update(probs, counts, idx, weights, wsel, st, MASK_NMODELS, pmix, bit, lr, dt)


# ----------------------------------------------------------------------------
# Value model: quantisation indices q in [0, nsym), coded as a binary tree
# (MSB first) with contexts from causal neighbours.
# ----------------------------------------------------------------------------
VAL_NMODELS = 7
VAL_TABLE_BITS = 22


@njit(cache=True)
def _val_neighbours(q, m, t, i, j, T, Y, X, nb):
    # returns neighbour symbols or nsym (missing) -- encoded as value or -1
    def g(tt, ii, jj):
        if ii < 0 or ii >= Y or jj < 0 or jj >= X or tt < 0:
            return -1
        if m[tt, ii, jj] == 1:
            return -1
        return q[tt, ii, jj]

    nb[0] = g(t, i, j - 1)  # L
    nb[1] = g(t, i - 1, j)  # U
    nb[2] = g(t, i - 1, j - 1)  # UL
    nb[3] = g(t, i - 1, j + 1)  # UR
    nb[4] = g(t, i, j - 2)  # LL
    nb[5] = g(t, i - 2, j)  # UU
    nb[6] = g(t - 1, i, j)  # P (previous slice)
    nb[7] = g(t - 1, i, j + 1)  # PR
    nb[8] = g(t - 1, i + 1, j)  # PD
    nb[9] = g(t, i - 1, j + 2)  # URR


@njit(cache=True)
def _val_pred(nb):
    """Median-edge-detector style prediction from L, U, UL (falls back)."""
    L, U, UL = nb[0], nb[1], nb[2]
    if L >= 0 and U >= 0 and UL >= 0:
        mx = max(L, U)
        mn = min(L, U)
        if UL >= mx:
            return mn
        if UL <= mn:
            return mx
        return L + U - UL
    if L >= 0 and U >= 0:
        return (L + U + 1) // 2
    if L >= 0:
        return L
    if U >= 0:
        return U
    if nb[3] >= 0:
        return nb[3]
    if nb[6] >= 0:
        return nb[6]
    return -1


@njit(cache=True)
def _code_values(q, m, T, Y, X, nbits, out, inp, state, encode, lr, lim):
    probs = np.full((VAL_NMODELS, 1 << VAL_TABLE_BITS), MODEL_ONE // 2, np.int32)
    counts = np.zeros((VAL_NMODELS, 1 << VAL_TABLE_BITS), np.uint8)
    nnodes = 1 << nbits
    weights = np.full((nnodes * 4, VAL_NMODELS + 1), 0.25, np.float64)
    idx = np.zeros(VAL_NMODELS, np.int64)
    st = np.zeros(VAL_NMODELS + 1, np.float64)
    nb = np.zeros(10, np.int64)
    dt = _make_dt(lim)
    tmask = (1 << VAL_TABLE_BITS) - 1
    for t in range(T):
        for i in range(Y):
            for j in range(X):
                if m[t, i, j] == 1:
                    continue
                _val_neighbours(q, m, t, i, j, T, Y, X, nb)
                L, U, UL, UR, LL, UU, P, PR, PD, URR = (
                    nb[0], nb[1], nb[2], nb[3], nb[4], nb[5], nb[6], nb[7], nb[8], nb[9],
                )
                pred = _val_pred(nb)
                # activity / texture context
                act = 0
                if L >= 0 and U >= 0:
                    act = abs(L - U)
                if UL >= 0 and U >= 0:
                    act += abs(UL - U)
                if UR >= 0 and U >= 0:
                    act += abs(UR - U)
                if L >= 0 and LL >= 0:
                    act += abs(L - LL)
                if act > 12:
                    act = 12
                dP = 0
                if P >= 0 and L >= 0:
                    dP = P - L
                    if dP > 6:
                        dP = 6
                    elif dP < -6:
                        dP = -6
                # weight-set selector: tree node + coarse activity
                asel = 0 if act == 0 else (1 if act <= 2 else (2 if act <= 5 else 3))
                node = 1
                sym = q[t, i, j] if encode else 0
                for b in range(nbits - 1, -1, -1):
                    idx[0] = _hash(L + 1, U + 1, node, 11, 0, tmask)
                    idx[1] = _hash(L + 1, U + 1, UL + 1, UR + 1, node * 8 + 12, tmask)
                    idx[2] = _hash(pred + 1, act, node, 13, 0, tmask)
                    idx[3] = _hash(L + 1, P + 1, node, 14, 0, tmask)
                    idx[4] = _hash(U + 1, P + 1, PR + 1, node, 15, tmask)
                    idx[5] = _hash(L + 1, LL + 1, U + 1, UU + 1, node * 8 + 16, tmask)
                    idx[6] = _hash(pred + 1, dP + 8, URR + 1, node, 17, tmask)
                    wsel = node * 4 + asel
                    pmix = _predict(probs, counts, idx, weights, wsel, st, VAL_NMODELS)
                    p12 = _to_p12(pmix)
                    if encode:
                        bit = (sym >> b) & 1
                        _enc_bit(state, out, p12, bit)
                    else:
                        bit = _dec_bit(state, inp, p12)
                    _update(probs, counts, idx, weights, wsel, st, VAL_NMODELS, pmix, bit, lr, dt)
                    node = node * 2 + bit
                if not encode:
                    q[t, i, j] = node - nnodes


@njit(cache=True)
def _encode_all(m, q, T, Y, X, nbits, out, lr_m, lim_m, lr_v, lim_v):
    state = np.zeros(5, np.int64)
    state[0] = 0
    state[1] = MASK32
    state[2] = 0
    state[3] = 1
    state[4] = 0
    dummy = np.zeros(1, np.uint8)
    _code_mask(m, T, Y, X, out, dummy, state, True, lr_m, lim_m)
    mask_end = state[4]
    _code_values(q, m, T, Y, X, nbits, out, dummy, state, True, lr_v, lim_v)
    _enc_flush(state, out)
    return state[4], mask_end


@njit(cache=True)
def _decode_all(inp, T, Y, X, nbits, lr_m, lim_m, lr_v, lim_v):
    state = np.zeros(5, np.int64)
    _dec_init(inp, state)
    m = np.zeros((T, Y, X), np.uint8)
    q = np.zeros((T, Y, X), np.int32)
    dummy = np.zeros(1, np.uint8)
    _code_mask(m, T, Y, X, dummy, inp, state, False, lr_m, lim_m)
    _code_values(q, m, T, Y, X, nbits, dummy, inp, state, False, lr_v, lim_v)
    return m, q


# ----------------------------------------------------------------------------
# numcodecs codec
# ----------------------------------------------------------------------------
class NanContextCodec(Codec):
    """
    Error-bounded (abs) codec preserving NaNs, using uniform quantisation with
    bin width 2*eb and a context-mixing arithmetic coder for the NaN mask and
    the quantisation indices.

    Parameters
    ----------
    eb_abs : float
        Absolute error bound. Reconstruction error is <= eb_abs for all finite
        values; NaN values are reconstructed as NaN.
    """

    codec_id = "nan-context-mixing"

    def __init__(self, eb_abs: float, lr_mask: float = 0.02, lim_mask: float = 1/512, lr_val: float = 0.015, lim_val: float = 1/256):
        self.eb_abs = float(eb_abs)
        self.lr_mask = float(lr_mask)
        self.lim_mask = float(lim_mask)
        self.lr_val = float(lr_val)
        self.lim_val = float(lim_val)
        self._last_mask_bytes = None

    def encode(self, buf):
        x = np.ascontiguousarray(buf)
        dtype = x.dtype
        shape = x.shape
        x3 = x.reshape((-1,) + shape[-2:]) if x.ndim >= 2 else x.reshape(1, 1, -1)
        T, Y, X = x3.shape
        m = np.isnan(x3).astype(np.uint8)
        finite = x3[m == 0].astype(np.float64)
        if finite.size == 0:
            xmin = 0.0
        else:
            xmin = float(finite.min())
        # bin width slightly below 2*eb so that rounding in the target dtype
        # can never exceed the bound
        step = 2.0 * self.eb_abs * (1.0 - 1e-6)
        q = np.zeros((T, Y, X), np.int32)
        vals = np.nan_to_num(x3.astype(np.float64), nan=xmin)
        q[:] = np.rint((vals - xmin) / step).astype(np.int32)
        q[m == 1] = 0
        qmax = int(q.max()) if finite.size else 0
        nbits = max(1, int(qmax).bit_length())
        out = np.zeros(x3.size * 2 + 1024, np.uint8)
        n, mask_end = _encode_all(m, q, T, Y, X, nbits, out, self.lr_mask, self.lim_mask, self.lr_val, self.lim_val)
        self._last_mask_bytes = int(mask_end)
        header = struct.pack(
            "<4sBBddddddd",
            b"NCM1",
            len(shape),
            nbits,
            xmin,
            step,
            float(self.eb_abs),
            self.lr_mask,
            self.lim_mask,
            self.lr_val,
            self.lim_val,
        )
        header += struct.pack("<%dq" % len(shape), *shape)
        header += dtype.str.encode().ljust(8, b"\0")
        return header + out[:n].tobytes()

    def decode(self, buf, out=None):
        buf = memoryview(buf).tobytes()
        magic, ndim, nbits, xmin, step, eb, lr_m, lim_m, lr_v, lim_v = struct.unpack("<4sBBddddddd", buf[:62])
        assert magic == b"NCM1"
        off = 62
        shape = struct.unpack("<%dq" % ndim, buf[off : off + 8 * ndim])
        off += 8 * ndim
        dtype = np.dtype(buf[off : off + 8].rstrip(b"\0").decode())
        off += 8
        inp = np.frombuffer(buf[off:], np.uint8)
        inp = np.concatenate([inp, np.zeros(16, np.uint8)])
        shape3 = (int(np.prod(shape[:-2])), shape[-2], shape[-1]) if ndim >= 2 else (1, 1, shape[0])
        T, Y, X = shape3
        m, q = _decode_all(inp, T, Y, X, nbits, lr_m, lim_m, lr_v, lim_v)
        vals = xmin + step * q.astype(np.float64)
        vals[m == 1] = np.nan
        if np.issubdtype(dtype, np.integer):
            vals = np.rint(vals)
        result = vals.reshape(shape).astype(dtype)
        if out is not None:
            out[...] = result
            return out
        return result


register_codec(NanContextCodec)
