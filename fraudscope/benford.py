"""Digit distribution tests built on benford_py's probability formulas."""

import numpy as np
import pandas as pd
from scipy.stats import chi2

from benford.digits import expected_probs, extract_digits
from benford.constants import MAD_CONFORM

from .applicability import benford_applicable
from .results import TestResult


def conformity(mad: float, mode: str) -> str:
    """Classify MAD using Nigrini (2012), *Benford's Law*, Table 3.1.

    First: .006/.012/.015; second: .008/.010/.012; first-two:
    .0012/.0018/.0022 (close/acceptable/marginal). No published
    last-two thresholds in that table: return 'not established'.
    These are descriptive guidelines, not calibrated significance tests.
    """
    limits = MAD_CONFORM[{"first": 1, "second": 22, "first_two": 2, "last_two": -2}[mode]]
    if limits is None:
        return "not established"
    return next((label for bound, label in zip(limits, ("close", "acceptable", "marginal"))
                 if mad < bound), "nonconformity")


def benford_test(series: pd.Series, mode: str = "first", *, column: str | None = None,
                 top_k: int = 3, decimals: int = 2, strict: bool = False) -> TestResult:
    """Test multinomial bins: chi-square=sum((O-E)^2/E), MAD=mean(|p-q|).

    Z for each bin uses the binomial standard deviation and continuity
    correction max(0, |O-E|-0.5)/sqrt(n*q*(1-q)). Flags are input positions
    belonging to the highest positively deviating bins, not evidence against
    individual records. Chi-square's approximation needs adequate expected counts.
    """
    if top_k < 0:
        raise ValueError("top_k must be nonnegative")
    series = pd.Series(series).reset_index(drop=True)
    _, warnings = benford_applicable(series, strict=strict)
    digits = extract_digits(series, mode, decimals)
    bins, expected = expected_probs(mode)
    n = len(digits)
    counts = digits.value_counts().reindex(bins, fill_value=0).to_numpy()
    observed = counts / n if n else np.zeros(len(bins))
    mad = float(np.mean(np.abs(observed - expected))) if n else None
    chi_square = float(np.sum((counts - n * expected)**2 / (n * expected))) if n else 0.0
    p_value = float(chi2.sf(chi_square, len(bins) - 1)) if n else None
    z = np.maximum(0, np.abs(counts - n * expected) - 0.5) / np.sqrt(
        n * expected * (1 - expected)) if n else np.zeros(len(bins))
    positive = np.where(observed > expected)[0]
    chosen = positive[np.argsort(-z[positive], kind="stable")[:top_k]]
    chosen_bins = bins[chosen]
    if n * expected.min() < 5:
        warnings.append("Some expected bin counts are below five; chi-square calibration is unreliable.")
    if not n:
        warnings.append("No eligible positive values for this digit mode.")
    details = {"mode": mode, "bins": bins.tolist(), "observed": observed.tolist(),
               "expected": expected.tolist(), "counts": counts.tolist(),
               "z_scores": z.tolist(), "chi_square": chi_square,
               "chi_square_p": p_value, "mad": mad,
               "conformity": conformity(mad, mode) if mad is not None else "not established",
               "flagged_bins": chosen_bins.tolist(), "decimals": decimals}
    return TestResult("benford_" + mode, column or series.name, chi_square,
                      p_value, "asymptotic" if n else "none", mad, n, details,
                      digits.index[digits.isin(chosen_bins)].astype(int).tolist(),
                      ["Bounded or size-limited data may not follow this distribution.",
                       "Rounding, price thresholds and data-entry conventions can shift digit frequencies.",
                       "At large sample sizes chi-square detects tiny deviations; MAD is the primary measure."],
                      warnings)
