"""Matplotlib figures shared by the frontend and portable reports."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from fraudscope.plots import (
    plot_ecdf_vs_reference as _plot_ecdf_vs_reference,
)
from fraudscope.plots import (
    plot_group_residuals as _plot_group_residuals,
)
from fraudscope.plots import (
    plot_residuals as _plot_residuals,
)
from fraudscope.plots import (
    plot_summary_dashboard as _plot_summary_dashboard,
)

from .typography import style_figure

OBSERVED = "#287d9b"
EXPECTED = "#a34b56"
FLAGGED = "#d37a32"


def digit_figure(result, view="primary", *, confidence=0.95):
    """Reference band per digit: expected +/- z*binomial SE +/- 0.5/n.

    ``confidence`` controls only the plotted range, never test p-values,
    Findings, or positional record flags. Bands are per bin, not simultaneous.
    """
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("Chart band must be a fraction strictly between 0 and 1.")
    d = result.details
    bins = np.asarray(d["bins"])
    observed, expected = np.asarray(d["observed"]), np.asarray(d["expected"])
    x = np.arange(len(bins))
    critical = norm.ppf((1 + confidence) / 2)
    sd = np.sqrt(expected * (1 - expected) / max(result.n, 1))
    margin = critical * sd + 0.5 / max(result.n, 1)
    lower = np.maximum(0, expected - margin)
    upper = np.minimum(1, expected + margin)
    outside = (observed < lower) | (observed > upper)
    label = f"{confidence:.0%} reference band"
    fig, ax = plt.subplots(figsize=(12, 6.5))
    if view == "deviation":
        ax.bar(
            x,
            observed - expected,
            color=np.where(outside, FLAGGED, OBSERVED),
        )
        ax.plot(x, upper - expected, "--", color=EXPECTED, label=label)
        ax.plot(x, lower - expected, "--", color=EXPECTED)
        ax.set_ylabel("Seen in data − expected")
    elif view in ("primary", "log"):
        ax.bar(
            x, observed, color=np.where(outside, FLAGGED, OBSERVED), label="Observed"
        )
        ax.plot(
            x,
            expected,
            color=EXPECTED,
            marker="o",
            markersize=2,
            linewidth=1.2,
            label="Expected",
        )
        ax.fill_between(x, lower, upper, color=EXPECTED, alpha=0.15, label=label)
        if view == "log":
            ax.set_yscale("log")
            ax.set_ylim(bottom=max(1e-5, min(expected) * 0.1))
        ax.set_ylabel("Share of values")
    else:
        raise ValueError("Unknown digit figure view")
    ticks = np.unique(np.linspace(0, len(x) - 1, min(20, len(x)), dtype=int))
    ax.set_xticks(
        ticks,
        [f"{bins[i]:02d}" if d["mode"] == "last_two" else str(bins[i]) for i in ticks],
    )
    ax.set(
        xlabel="Digit group",
        title=f"{d['mode'].replace('_', ' ').title()} digits · Average gap (MAD) {d['mad']:.4g} · {d['conformity']}",
    )
    ax.legend(loc="best")
    fig.tight_layout()
    return style_figure(fig)


def mantissa_figure(values):
    """Plot fractional log10 of eligible positive amounts against uniform density."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    fig, ax = plt.subplots(figsize=(9, 4))
    if len(values):
        ax.hist(np.mod(np.log10(values), 1), bins=20, density=True, color=OBSERVED)
    ax.axhline(1, color=EXPECTED, label="Uniform reference")
    ax.set(
        xlabel="Fractional log10(value)",
        ylabel="Density",
        title="Log-mantissa distribution",
    )
    ax.legend()
    fig.tight_layout()
    return style_figure(fig)


def score_figure(scores, threshold=0.9):
    """Histogram of relative record rankings."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores, bins=20, range=(0, 1), color=OBSERVED)
    ax.axvline(threshold, color=FLAGGED, label="Review threshold")
    ax.set(
        xlabel="Relative anomaly score",
        ylabel="Records",
        title="Anomaly score distribution",
    )
    ax.legend()
    fig.tight_layout()
    return style_figure(fig)


def top_records_figure(scores, contributions, n=10):
    """Highest-scoring row positions and their strongest contributing feature."""
    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.argsort(-np.asarray(scores))[:n][::-1]
    ax.barh(
        [f"Row {p}" for p in positions], np.asarray(scores)[positions], color=FLAGGED
    )
    for i, pos in enumerate(positions):
        explanation = contributions.get(int(pos), [])
        if explanation:
            ax.text(
                float(scores[pos]) + 0.01, i, explanation[0], va="center", fontsize=8
            )
    ax.set(
        xlabel="Relative anomaly score", xlim=(0, 1.3), title="Top records to review"
    )
    fig.tight_layout()
    return style_figure(fig)


def plot_ecdf_vs_reference(result, save_path=None):
    return style_figure(_plot_ecdf_vs_reference(result, save_path))


def plot_group_residuals(result, save_path=None):
    return style_figure(_plot_group_residuals(result, save_path))


def plot_residuals(result, save_path=None):
    return style_figure(_plot_residuals(result, save_path))


def plot_summary_dashboard(results, save_path=None):
    return style_figure(_plot_summary_dashboard(results, save_path))


__all__ = [
    "digit_figure",
    "mantissa_figure",
    "plot_ecdf_vs_reference",
    "plot_group_residuals",
    "plot_residuals",
    "plot_summary_dashboard",
    "score_figure",
    "top_records_figure",
]
