"""Assemble numerical sections, chart images, and Findings-based readings."""

import base64
import io

import matplotlib.pyplot as plt

from .figures import (
    digit_figure,
    plot_ecdf_vs_reference,
    plot_group_residuals,
    plot_residuals,
    plot_summary_dashboard,
    score_figure,
    top_records_figure,
)


def _image(fig):
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    finally:
        plt.close(fig)


def build_report(bundle, *, top_n=20):
    """Create portable report data; Findings alone supply written readings."""
    sections = []
    for i, result in enumerate(bundle.results):
        if result.p_value is None:
            continue
        name = result.test_name
        numbers = {
            "n": result.n,
            "statistic": result.statistic,
            "p_value": result.p_value,
            "p_value_method": result.p_value_method,
            "adjusted_p": result.details.get("p_adjusted"),
            "effect_size": result.effect_size,
        }
        charts = []
        if name.startswith("benford_"):
            numbers.update(
                {
                    key: result.details[key]
                    for key in (
                        "conformity",
                        "bins",
                        "observed",
                        "expected",
                        "z_scores",
                    )
                }
            )
            for view in ("primary", "log", "deviation"):
                charts.append(
                    (
                        view,
                        _image(digit_figure(result, view)),
                        "Compare observed bars with the expected line and reference bounds.",
                    )
                )
            if f"{i}:mantissa" in bundle.figures:
                charts.append(
                    (
                        "mantissa",
                        _image(bundle.figures[f"{i}:mantissa"]()),
                        "Compare fractional log-magnitudes with the uniform reference.",
                    )
                )
        elif name in ("ks_one_sample", "kuiper_one_sample"):
            numbers.update(
                {
                    key: result.details[key]
                    for key in ("D_plus", "D_minus", "x_plus", "x_minus")
                }
            )
            charts.append(
                (
                    "ecdf",
                    _image(plot_ecdf_vs_reference(result)),
                    "Compare the sample's cumulative curve with its reference.",
                )
            )
        elif name == "regression":
            numbers.update(
                {
                    key: result.details[key]
                    for key in (
                        "r_squared",
                        "adj_r_squared",
                        "breusch_pagan",
                        "normality",
                    )
                }
            )
            numbers["flagged_rows"] = len(result.flagged_indices)
            charts.append(
                (
                    "residuals",
                    _image(plot_residuals(result)),
                    "Inspect residual shape and points with high influence.",
                )
            )
            if "groups" in result.details:
                charts.append(
                    (
                        "groups",
                        _image(plot_group_residuals(result)),
                        "Compare residual distributions across groups.",
                    )
                )
        section_findings = [
            f
            for f in bundle.findings
            if f.section == name and f.column == result.column
        ]
        sections.append(
            {
                "title": f"{name.replace('_', ' ').title()} · {result.column}",
                "numbers": numbers,
                "charts": charts,
                "reading": [f.statement for f in section_findings],
                "caveats": result.caveats,
                "warnings": result.warnings,
            }
        )
    ml = bundle.ml_output
    if ml is not None and ml.scores:
        sections.append(
            {
                "title": "Anomaly scoring",
                "numbers": {
                    "index": ml.index,
                    "components": ml.components,
                    "threshold": ml.threshold,
                    "reviewed_records": len(ml.flagged_indices),
                },
                "charts": [
                    (
                        "scores",
                        _image(score_figure(ml.scores, ml.threshold)),
                        "Higher values rank records as more unusual within this dataset.",
                    ),
                    (
                        "top",
                        _image(top_records_figure(ml.scores, ml.contributions, top_n)),
                        "Bars show the highest-scoring row positions and a contributing feature.",
                    ),
                ],
                "reading": [
                    f.statement
                    for f in bundle.findings
                    if f.section == "anomaly_scoring"
                ],
                "caveats": [
                    "Rankings are relative, not probabilities; collection and data-entry conventions may affect scores."
                ],
                "warnings": [],
            }
        )
    cover = _image(plot_summary_dashboard(bundle.results))
    severity_order = {"strong": 0, "notable": 1, "info": 2}
    summary = sorted(bundle.findings, key=lambda f: (severity_order[f.severity], f.id))[
        :10
    ]
    return {
        "metadata": bundle.run_metadata,
        "profile": bundle.dataset_profile,
        "cover": cover,
        "sections": sections,
        "summary": summary,
        "flagged_rows": sorted(
            {p for r in bundle.results for p in r.flagged_indices}
            | (set(ml.flagged_indices) if ml else set())
        )[:top_n],
        "index": ml.index if ml else None,
        "components": ml.components if ml else {},
    }
