"""Matplotlib figures from structured statistical results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def _save(fig, path):
    if path is not None:
        fig.savefig(Path(path), dpi=150, bbox_inches="tight")
    return fig


def plot_benford(result, save_path=None):
    """Observed digit proportions, expected values and 95% binomial bounds."""
    d = result.details
    bins, obs, exp = (np.asarray(d[k]) for k in ("bins", "observed", "expected"))
    fig, ax = plt.subplots(figsize=(11, 4))
    colors = ["#db773b" if b in d["flagged_bins"] else "#277da1" for b in bins]
    ax.bar(np.arange(len(bins)), obs, color=colors, label="Observed")
    ax.plot(np.arange(len(bins)), exp, color="#a62438", label="Expected")
    sd = np.sqrt(exp * (1 - exp) / max(1, result.n))
    ax.fill_between(np.arange(len(bins)), np.maximum(0, exp - 1.96 * sd),
                    exp + 1.96 * sd, color="#a62438", alpha=0.18, label="Expected ± 1.96 SD")
    ticks = np.linspace(0, len(bins) - 1, min(len(bins), 20), dtype=int)
    ax.set_xticks(ticks, [f"{int(bins[i]):02d}" if d["mode"] == "last_two" else str(bins[i]) for i in ticks])
    ax.set(xlabel="Digit bin", ylabel="Proportion", title="Digit distribution deviation")
    ax.legend()
    return _save(fig, save_path)


def plot_ecdf_vs_reference(result, save_path=None):
    """ECDF with reference curve; mark the D+ and D- locations."""
    d = result.details
    x = np.asarray(d["sample"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", label="Sample ECDF")
    if "other_sample" in d:
        other = np.asarray(d["other_sample"])
        ax.step(other, np.arange(1, len(other) + 1) / len(other), where="post", label="Comparison ECDF")
    else:
        ax.plot(x, d["reference_cdf"], label="Reference CDF")
    for key, color in (("x_plus", "#d55e00"), ("x_minus", "#9955aa")):
        ax.axvline(d[key], color=color, alpha=0.65, linestyle="--", label=key.replace("x_", "D"))
    ax.set(xlabel="Value", ylabel="Cumulative probability", title="ECDF deviation from reference")
    ax.legend()
    return _save(fig, save_path)


def plot_residuals(result, save_path=None):
    """Residual/fitted, QQ, Cook distance and leverage/studentized diagnostics."""
    d = result.details
    fitted, resid, cook, leverage, student = (np.asarray(d[key]) for key in
        ("fitted", "residuals", "cooks_distance", "leverage", "studentized_residuals"))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].scatter(fitted, resid, s=12)
    axes[0, 0].axhline(0, color="gray")
    axes[0, 0].set(xlabel="Fitted", ylabel="Residual", title="Residuals vs fitted")
    stats.probplot(resid, plot=axes[0, 1])
    axes[0, 1].set_title("Residual QQ")
    axes[1, 0].vlines(d["row_positions"], 0, cook)
    axes[1, 0].set(xlabel="Row position", ylabel="Cook's distance", title="Influence")
    axes[1, 1].scatter(leverage, student, s=12)
    axes[1, 1].set(xlabel="Leverage", ylabel="Studentized residual", title="Leverage and residuals")
    for pos in result.flagged_indices:
        match = d["row_positions"].index(pos)
        axes[1, 1].annotate(str(pos), (leverage[match], student[match]), fontsize=8)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_group_residuals(result, save_path=None):
    """Boxplots of regression residuals by group."""
    groups = result.details["groups"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.boxplot([d["residuals"] for d in groups.values()], tick_labels=list(groups))
    ax.set(xlabel="Group", ylabel="Residual", title="Residual distribution by group")
    return _save(fig, save_path)


def plot_summary_dashboard(results, save_path=None):
    """Compact grid of statistics and adjusted significance for a run."""
    fig, axes = plt.subplots(max(1, len(results)), 1, figsize=(9, max(3, 2.3 * len(results))), squeeze=False)
    for ax, result in zip(axes.flat, results):
        ax.axis("off")
        ax.text(0.02, 0.7, f"{result.test_name}: {result.column or 'comparison'}", fontsize=11)
        statistic = f"{result.statistic:.4g}" if isinstance(result.statistic, (int, float)) else str(result.statistic)
        ax.text(0.02, 0.3, f"n={result.n}   statistic={statistic}   "
                f"p={result.p_value if result.p_value is not None else 'N/A'}   "
                f"adjusted p={result.details.get('p_adjusted', 'N/A')}", fontsize=9)
    fig.suptitle("Statistical anomaly overview")
    fig.tight_layout()
    return _save(fig, save_path)
