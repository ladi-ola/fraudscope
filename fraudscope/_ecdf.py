"""Shared empirical CDF computations and reference handling."""

import numpy as np
import pandas as pd

from .applicability import continuous_applicable


def sample(series, strict=False):
    _, warnings = continuous_applicable(series, strict=strict)
    values = pd.to_numeric(pd.Series(series).reset_index(drop=True), errors="coerce")
    values = values[np.isfinite(values.to_numpy(dtype=float))]
    if not len(values):
        raise ValueError("No finite numeric observations.")
    return values, warnings


def reference_data(values, reference):
    """'benford' maps positive x to fractional log10(x), uniform on [0,1)."""
    if isinstance(reference, str):
        if reference != "benford":
            raise ValueError("Unknown reference distribution.")
        values = values[values > 0]
        if len(values) == 0:
            raise ValueError("Benford mantissas require positive observations.")
        transformed = np.mod(np.log10(values.to_numpy(dtype=float)), 1)
        from scipy.stats import uniform
        return pd.Series(transformed, index=values.index), uniform.cdf, "uniform", True
    if hasattr(reference, "cdf"):
        return values, reference.cdf, repr(reference), True
    if callable(reference):
        return values, reference, getattr(reference, "__name__", "callable CDF"), True
    # A finite (bins, probabilities) reference is discrete.
    bins, probabilities = reference
    bins = np.asarray(bins, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if (bins.ndim != 1 or len(bins) != len(probabilities) or len(bins) < 2 or
            not np.all(np.diff(bins) > 0) or np.any(probabilities < 0) or
            not np.isclose(probabilities.sum(), 1)):
        raise ValueError("Discrete reference requires sorted bins and normalized probabilities.")
    def cdf(x):
        return np.r_[0, np.cumsum(probabilities)][np.searchsorted(bins, x, side="right")]
    return values, cdf, "discrete", False


def one_sample_d(x, cdf, discrete=False):
    """D+=max(i/n-F(x_i)); D-=max(F(x_i)-(i-1)/n)."""
    ordered = np.sort(np.asarray(x, dtype=float))
    n = len(ordered)
    f = np.asarray(cdf(ordered), dtype=float)
    if f.shape == ():
        f = np.array([cdf(v) for v in ordered])
    if f.shape != ordered.shape or not np.all(np.isfinite(f)):
        raise ValueError("CDF must return finite probabilities per observation.")
    plus = np.arange(1, n + 1) / n - f
    if discrete:
        unique, starts, counts = np.unique(ordered, return_index=True, return_counts=True)
        left = np.maximum(0, np.asarray(cdf(np.nextafter(unique, -np.inf)), dtype=float))
        minus = np.repeat(left - starts / n, counts)
    else:
        minus = f - np.arange(n) / n
    ip, im = np.argmax(plus), np.argmax(minus)
    return float(max(plus[ip], 0)), float(max(minus[im], 0)), float(ordered[ip]), float(ordered[im])


def two_sample_d(a, b):
    """Evaluate both right-continuous ECDFs at every unique observation."""
    a, b = np.sort(a), np.sort(b)
    points = np.union1d(a, b)
    difference = (np.searchsorted(a, points, side="right") / len(a) -
                  np.searchsorted(b, points, side="right") / len(b))
    ip, im = np.argmax(difference), np.argmin(difference)
    return float(max(0, difference[ip])), float(max(0, -difference[im])), float(points[ip]), float(points[im])


def mc_one(x, bins, probabilities, observed, statistic, n_sim, random_state):
    if n_sim < 1:
        raise ValueError("n_sim must be positive")
    rng = np.random.default_rng(random_state)
    cumsum = np.r_[0, np.cumsum(probabilities)]
    def cdf(v):
        return cumsum[np.searchsorted(bins, v, side="right")]
    exceed = 0
    for _ in range(n_sim):
        draw = rng.choice(bins, size=len(x), p=probabilities)
        dp, dm, _, _ = one_sample_d(draw, cdf, discrete=True)
        exceed += statistic(dp, dm) >= observed - 1e-14
    return (1 + exceed) / (1 + n_sim)


def details(values, cdf, label, dp, dm, xp, xm):
    ordered = np.sort(values.to_numpy(dtype=float))
    evaluated = np.asarray(cdf(ordered))
    if evaluated.shape == ():
        evaluated = np.asarray([cdf(v) for v in ordered])
    return {"sample": ordered.tolist(), "reference_cdf": evaluated.tolist(),
            "reference": label, "D_plus": dp, "D_minus": dm,
            "x_plus": xp, "x_minus": xm}
