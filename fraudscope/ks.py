"""Vectorized one- and two-sample Kolmogorov-Smirnov tests."""

import numpy as np
from scipy import stats

from ._ecdf import details, mc_one, one_sample_d, reference_data, sample, two_sample_d
from .results import TestResult


def ks_one_sample(
    series,
    reference,
    *,
    n_sim: int = 10_000,
    random_state=None,
    strict: bool = False,
    column=None,
) -> TestResult:
    """Test a CDF, scipy frozen distribution, Benford mantissas, or (bins, probs).

    'benford' transforms positive x to fractional log10(x) in [0,1), whose
    reference CDF is uniform. Discrete references use simulated null samples.
    """
    values, warnings = sample(series, strict)
    values, cdf, label, continuous = reference_data(values, reference)
    if not continuous:
        bins, probs = map(np.asarray, reference)
        if not np.isin(values, bins).all():
            warnings.append(
                "Some observations are outside the discrete reference bins."
            )
    dp, dm, xp, xm = one_sample_d(values, cdf, discrete=not continuous)
    d = max(dp, dm)
    if continuous:
        p = float(stats.ks_1samp(values.to_numpy(), cdf).pvalue)
        method = "asymptotic"  # scipy chooses finite-n exact where possible
    else:
        p = mc_one(values, bins, probs, d, max, n_sim, random_state)
        method = "monte_carlo"
    return TestResult(
        "ks_one_sample",
        column or getattr(series, "name", None),
        d,
        p,
        method,
        d,
        len(values),
        details(values, cdf, label, dp, dm, xp, xm),
        [],
        [
            "Discreteness, rounding, or fitted parameters alter null calibration.",
            "Sample selection can explain deviations from a reference CDF.",
        ],
        warnings,
    )


def ks_two_sample(a, b, *, strict=False) -> TestResult:
    """Compare two independent ECDFs; D=max(D+,D-)."""
    left, w1 = sample(a, strict)
    right, w2 = sample(b, strict)
    dp, dm, xp, xm = two_sample_d(left.to_numpy(), right.to_numpy())
    d = max(dp, dm)
    p = float(stats.ks_2samp(left, right).pvalue)
    return TestResult(
        "ks_two_sample",
        getattr(a, "name", None),
        d,
        p,
        "asymptotic",
        d,
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
        ["Different sampling periods or population mixtures can shift distributions."],
        w1 + w2,
    )
