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
    if values.empty:
        reasons.append("No finite numeric observations are available.")
    elif values.nunique() == 1:
        reasons.append("Constant column; continuous reference is unsuitable.")
    if strict and reasons:
        raise ValueError(" ".join(reasons))
    return not reasons, reasons


def benford_applicable(series, strict: bool = False) -> tuple[bool, list[str]]:
    values = numeric_values(series)
    positives = values[values > 0]
    reasons = []
    if positives.empty:
        reasons.append("No positive finite observations are available.")
    elif len(positives) < len(values):
        reasons.append("Nonpositive values were excluded from digit analysis.")
    if len(positives) > 1 and positives.nunique() == 1:
        reasons.append(
            "All positive values are identical; a digit comparison is uninformative."
        )
    elif len(positives) > 1 and (
        (len(positives) - positives.nunique()) / len(positives) > 0.1
    ):
        reasons.append("Many repeated values; rounding or conventions may dominate.")
    if len(positives) > 1 and (
        np.mean(np.isclose(positives.to_numpy() % 100, 0, atol=1e-8)) > 0.5
    ):
        reasons.append(
            "Many values are multiples of 100; round-value conventions may dominate."
        )
    if len(positives) > 1:
        arr = np.sort(positives.to_numpy())
        lengths = np.floor(np.log10(arr)).astype(int)
        if len(np.unique(lengths)) == 1 and np.mean(np.diff(arr) == 1) > 0.7:
            reasons.append("Fixed-width near-sequential values resemble identifiers.")
    if strict and reasons:
        raise ValueError(" ".join(reasons))
    return not reasons, reasons
