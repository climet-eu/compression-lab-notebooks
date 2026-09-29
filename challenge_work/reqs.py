"""Fast (float64) mirror of the compression_requirement_checks semantics plus
requirement structure analysis used to derive search strategies."""

from dataclasses import dataclass, field

import numpy as np
from compression_recommendations.requirements.combinators import AllRequirements, AnyRequirement
from compression_recommendations.requirements.error_bounds.max import (
    MaxPointwiseAbsoluteErrorBoundRequirement,
    MaxPointwiseRangeRelativeErrorBoundRequirement,
    MaxPointwiseRelativeErrorBoundRequirement,
)
from compression_recommendations.requirements.error_bounds.mean import (
    MeanAbsoluteErrorBoundRequirement,
    MeanRangeRelativeErrorBoundRequirement,
    MeanRelativeErrorBoundRequirement,
)
from compression_recommendations.requirements.isovalue import IsovalueRequirement
from compression_recommendations.requirements.limits import DataLimitsRequirement
from compression_recommendations.requirements.lossless import LosslessRequirement
from compression_recommendations.requirements.missing import MissingValueRequirement

SAFETY = 1.0 - 1e-9  # conservative factor applied to bounds in the fast check


def _finite_or_zero(a):
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def fast_ok(o, r, req):
    """Pointwise ok-array mirroring compression_requirement_checks."""
    o = np.asarray(o, np.float64)
    r = np.asarray(r, np.float64)
    if isinstance(req, AnyRequirement):
        ok = np.zeros(o.shape, bool)
        for q in req.requirements:
            ok |= fast_ok(o, r, q)
        return ok
    if isinstance(req, AllRequirements):
        ok = np.ones(o.shape, bool)
        for q in req.requirements:
            ok &= fast_ok(o, r, q)
        return ok
    of, rf = _finite_or_zero(o), _finite_or_zero(r)
    err = np.abs(of - rf)
    same = (o == r) | (np.isnan(o) & np.isnan(r))
    if isinstance(req, MaxPointwiseAbsoluteErrorBoundRequirement):
        return (err <= req.value * SAFETY) | same
    if isinstance(req, MaxPointwiseRelativeErrorBoundRequirement):
        return (err <= np.abs(of) * req.value * SAFETY) | same
    if isinstance(req, MaxPointwiseRangeRelativeErrorBoundRequirement):
        fin = np.isfinite(o)
        if not fin.any():
            return same
        rng = float(o[fin].max()) - float(o[fin].min())
        return (err <= req.value * rng * SAFETY) | same
    if isinstance(req, (MeanAbsoluteErrorBoundRequirement, MeanRelativeErrorBoundRequirement, MeanRangeRelativeErrorBoundRequirement)):
        fin = np.isfinite(o)
        err_sum = float(err[fin].sum())
        if isinstance(req, MeanAbsoluteErrorBoundRequirement):
            bound = req.value * int(fin.sum())
        elif isinstance(req, MeanRelativeErrorBoundRequirement):
            bound = req.value * float(np.abs(of[fin]).sum())
        else:
            if not fin.any():
                return same
            rng = float(o[fin].max()) - float(o[fin].min())
            bound = req.value * rng * int(fin.sum())
        ok = np.full(o.shape, err_sum <= bound * SAFETY)
        isinf = np.isinf(o)
        ok[isinf] = (o == r)[isinf]
        isnan = np.isnan(o)
        ok[isnan] = np.isnan(r)[isnan]
        if isinstance(req, MeanRelativeErrorBoundRequirement):
            z = o == 0
            ok[z] = (r == 0)[z]
        return ok
    if isinstance(req, DataLimitsRequirement):
        ok = np.ones(o.shape, bool)
        if req.minimum is not None and req.maximum is not None:
            w = (o >= req.minimum) & (o <= req.maximum)
            ok[w] = ((r >= req.minimum) & (r <= req.maximum))[w]
        elif req.minimum is not None:
            w = o >= req.minimum
            ok[w] = (r >= req.minimum)[w]
        elif req.maximum is not None:
            w = o <= req.maximum
            ok[w] = (r <= req.maximum)[w]
        return ok
    if isinstance(req, LosslessRequirement):
        return o.view(np.uint64) == r.view(np.uint64)
    if isinstance(req, IsovalueRequirement):
        v = req.value
        ok = np.ones(o.shape, bool)
        w = o < v
        ok[w] = (r < v)[w]
        w = o == v
        ok[w] = (r == v)[w]
        w = o > v
        ok[w] = (r > v)[w]
        return ok
    if isinstance(req, MissingValueRequirement):
        v = req.value
        if np.isnan(v):
            return np.isnan(o) == np.isnan(r)
        return (o == v) == (r == v)
    raise TypeError(f"unknown requirement {type(req)}")


def fast_check(o, r, requirements):
    return all(bool(np.all(fast_ok(o, r, q))) for q in requirements)


@dataclass
class ReqInfo:
    lossless: bool = False
    max_abs: list = field(default_factory=list)
    max_rel: list = field(default_factory=list)
    max_range_rel: list = field(default_factory=list)
    mean_abs: list = field(default_factory=list)
    mean_rel: list = field(default_factory=list)
    mean_range_rel: list = field(default_factory=list)
    minimum: float | None = None
    maximum: float | None = None
    isovalue: bool = False
    missing: bool = False
    # (abs, rel) pairs appearing together in an Any -> abs-or-rel transform
    abs_or_rel: list = field(default_factory=list)

    @property
    def has_pointwise(self):
        return bool(self.max_abs or self.max_rel or self.max_range_rel)

    @property
    def has_mean(self):
        return bool(self.mean_abs or self.mean_rel or self.mean_range_rel)


def analyse(requirements):
    info = ReqInfo()

    def walk(req):
        if isinstance(req, AnyRequirement):
            leaves = list(req.requirements)
            absv = [q.value for q in leaves if isinstance(q, MaxPointwiseAbsoluteErrorBoundRequirement)]
            relv = [q.value for q in leaves if isinstance(q, MaxPointwiseRelativeErrorBoundRequirement)]
            if absv and relv:
                info.abs_or_rel.append((max(absv), max(relv)))
            for q in leaves:
                walk(q)
        elif isinstance(req, AllRequirements):
            for q in req.requirements:
                walk(q)
        elif isinstance(req, MaxPointwiseAbsoluteErrorBoundRequirement):
            info.max_abs.append(req.value)
        elif isinstance(req, MaxPointwiseRelativeErrorBoundRequirement):
            info.max_rel.append(req.value)
        elif isinstance(req, MaxPointwiseRangeRelativeErrorBoundRequirement):
            info.max_range_rel.append(req.value)
        elif isinstance(req, MeanAbsoluteErrorBoundRequirement):
            info.mean_abs.append(req.value)
        elif isinstance(req, MeanRelativeErrorBoundRequirement):
            info.mean_rel.append(req.value)
        elif isinstance(req, MeanRangeRelativeErrorBoundRequirement):
            info.mean_range_rel.append(req.value)
        elif isinstance(req, DataLimitsRequirement):
            info.minimum = req.minimum
            info.maximum = req.maximum
        elif isinstance(req, LosslessRequirement):
            info.lossless = True
        elif isinstance(req, IsovalueRequirement):
            info.isovalue = True
        elif isinstance(req, MissingValueRequirement):
            info.missing = True

    for r in requirements:
        walk(r)
    return info
