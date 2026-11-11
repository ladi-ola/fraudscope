"""Kuiper statistics for cyclic and continuous or discrete data."""

import numpy as np

from ._ecdf import details, mc_one, one_sample_d, reference_data, sample, two_sample_d
from .results import TestResult


def kuiper_sf(v: float, n: float) -> float:
    """Asymptotic Q_K((sqrt(n)+.155+.24/sqrt(n))V), 100 series terms.

    For tiny lambda the series suffers cancellation; its limiting value is 1.
    """
    lam = (np.sqrt(n) + 0.155 + 0.24 / np.sqrt(n)) * v
    if lam < 0.4:
        return 1.0
    j = np.arange(1, 101)
    q = 2 * np.sum((4 * j**2 * lam**2 - 1) * np.exp(-2 * j**2 * lam**2))
    return float(np.clip(q, 0, 1))


def kuiper_one_sample(
    series,
    reference,
    *,
    n_sim: int = 10_000,
    random_state=None,
    monte_carlo: bool = False,
    strict: bool = False,
    column=None,
) -> TestResult:
    """V=D+ + D-; useful for circular/time-of-day data and tail/shift changes.

    Unlike KS, a rotation of the origin on a circle does not change V.
    For discrete references supply (sorted bins, probabilities); Monte Carlo
    samples preserve jumps in the null CDF.
    """
    values, warnings = sample(series, strict)
    values, cdf, label, continuous = reference_data(values, reference)
    dp, dm, xp, xm = one_sample_d(values, cdf, discrete=not continuous)
    v = dp + dm
    if not continuous or monte_carlo:
        if n_sim < 1:
            raise ValueError("n_sim must be positive")
        if continuous:
            is_benford = isinstance(reference, str) and reference == "benford"
            if not hasattr(reference, "rvs") and not is_benford:
                raise ValueError(
                    "Monte Carlo requires a frozen scipy reference or 'benford'."
                )
            rng = np.random.default_rng(random_state)
            exceed = 0
            for _ in range(n_sim):
                draw = (
                    rng.uniform(size=len(values))
                    if is_benford
                    else reference.rvs(size=len(values), random_state=rng)
                )
                sp, sm, _, _ = one_sample_d(draw, cdf)
                exceed += sp + sm >= v - 1e-14
            p = (1 + exceed) / (1 + n_sim)
        else:
            bins, probs = map(np.asarray, reference)
            p = mc_one(values, bins, probs, v, lambda p, m: p + m, n_sim, random_state)
        method = "monte_carlo"
    else:
        p = kuiper_sf(v, len(values))
        method = "asymptotic"
    return TestResult(
        "kuiper_one_sample",
        column or getattr(series, "name", None),
        v,
        float(p),
        method,
        v,
        len(values),
        details(values, cdf, label, dp, dm, xp, xm),
        [],
        [
            "Rounding or restricted ranges can explain departures from the reference.",
            "The asymptotic approximation is less reliable at small sample sizes.",
        ],
        warnings,
    )


def kuiper_two_sample(
    a, b, *, n_sim: int = 10_000, random_state=None, monte_carlo=False, strict=False
) -> TestResult:
    """Two-sample V with effective n=n1*n2/(n1+n2); optional permutation p."""
    left, w1 = sample(a, strict)
    right, w2 = sample(b, strict)
    dp, dm, xp, xm = two_sample_d(left.to_numpy(), right.to_numpy())
    v = dp + dm
    if monte_carlo:
        if n_sim < 1:
            raise ValueError("n_sim must be positive")
        rng = np.random.default_rng(random_state)
        pooled = np.r_[left.to_numpy(), right.to_numpy()]
        exceed = 0
        for _ in range(n_sim):
            shuffled = rng.permutation(pooled)
            sp, sm, _, _ = two_sample_d(shuffled[: len(left)], shuffled[len(left) :])
            exceed += sp + sm >= v - 1e-14
        p, method = (1 + exceed) / (1 + n_sim), "monte_carlo"
    else:
        p, method = (
            kuiper_sf(v, len(left) * len(right) / (len(left) + len(right))),
            "asymptotic",
        )
    return TestResult(
        "kuiper_two_sample",
        getattr(a, "name", None),
        v,
        float(p),
        method,
        v,
        len(left) + len(right),
        {
            "D_plus": dp,
            "D_minus": dm,
            "x_plus": xp,
            "x_minus": xm,
            "sample": np.sort(left).tolist(),
            "other_sample": np.sort(right).tolist(),
            "n1": len(left),
            "n2": len(right),
        },
        [],
        ["Population mixtures or collection periods may explain a shift."],
        w1 + w2,
    )
