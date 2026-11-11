"""Heuristic suitability checks; failed checks warn unless strict is requested."""

import numpy as np
import pandas as pd


def numeric_values(series) -> pd.Series:
    """Return finite numeric observations while retaining original row labels."""
    s = pd.to_numeric(pd.Series(series), errors="coerce")
    return s.loc[np.isfinite(s.to_numpy(dtype=float))]


def continuous_applicable(series, strict: bool = False) -> tuple[bool, list[str]]:
    values = numeric_values(series)
    reasons = []
    if len(values) < 30:
        reasons.append("Fewer than 30 finite numeric observations.")
    if len(values) and values.nunique() == 1:
        reasons.append("Constant column; continuous reference is unsuitable.")
    if strict and reasons:
        raise ValueError(" ".join(reasons))
    return not reasons, reasons


def benford_applicable(series, strict: bool = False) -> tuple[bool, list[str]]:
    values = numeric_values(series)
    positives = values[values > 0]
    reasons = []
    if len(positives) < 300:
        reasons.append("Fewer than 300 positive observations.")
    elif len(positives) < 1000:
        reasons.append("Fewer than 1000 positives; estimates may be unstable.")
    if len(positives) == 0 or positives.max() / positives.min() < 1000:
        reasons.append("Positive values span fewer than three orders of magnitude.")
    if len(positives) and (
        positives.value_counts(normalize=True).iloc[0] > 0.1
        or positives.value_counts(normalize=True).head(5).sum() > 0.3
    ):
        reasons.append("Many repeated values; rounding or conventions may dominate.")
    if len(positives) >= 30 and (
        np.mean(np.isclose(positives.to_numpy() % 100, 0, atol=1e-8)) > 0.5
    ):
        reasons.append(
            "Many values are multiples of 100; round-value conventions may dominate."
        )
    if len(positives) >= 30:
        arr = np.sort(positives.to_numpy())
        lengths = np.floor(np.log10(arr)).astype(int)
        if len(np.unique(lengths)) == 1 and np.mean(np.diff(arr) == 1) > 0.7:
            reasons.append("Fixed-width near-sequential values resemble identifiers.")
    if strict and reasons:
        raise ValueError(" ".join(reasons))
    return not reasons, reasons
