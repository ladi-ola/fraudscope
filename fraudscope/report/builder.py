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
from .metrics import metric_explanation
from .typography import font_face_css, style_figure

NUMBER_LABELS = {
    "n": "Records used in this check (n)",
    "statistic": "Test result (statistic)",
    "p_value": "Result relative to the reference (p-value)",
    "p_value_method": "How the p-value was calculated",
    "adjusted_p": "P-value after accounting for multiple checks",
    "effect_size": "Size of the difference (effect size)",
    "finite_count": "Usable numeric values in this column",
    "positive_count": "Positive values in this column",
    "finite_min": "Smallest usable value",
    "finite_max": "Largest usable value",
    "positive_min": "Smallest positive value",
    "positive_max": "Largest positive value",
    "positive_orders_span": "Spread of positive values (powers of ten)",
    "ecdf_95_band": "95% sampling-uncertainty guide",
    "conformity": "Digit-pattern label",
    "reference_band": "Shaded digit-chart band",
    "D_plus": "Largest gap above the reference (D+)",
    "D_minus": "Largest gap below the reference (D-)",
    "x_plus": "Value at the largest upward gap",
    "x_minus": "Value at the largest downward gap",
    "r_squared": "Share of variation explained by the model (R²)",
    "adj_r_squared": "Model fit adjusted for predictors",
    "breusch_pagan": "Check for changing error spread",
    "normality": "Check of error shape",
    "flagged_rows": "Rows marked for review",
    "index": "Overall relative anomaly index",
    "components": "What goes into the index",
    "threshold": "Record-review score threshold",
    "reviewed_records": "Records above the review threshold",
}


def _title(result):
    if result.test_name.startswith("benford_"):
        return f"Benford {result.details['mode'].replace('_', ' ')} digits · {result.column}"
    if result.test_name == "ks_one_sample":
        return f"Distribution comparison (KS) · {result.column}"
    if result.test_name == "kuiper_one_sample":
        return f"Distribution comparison (Kuiper) · {result.column}"
    if result.test_name == "regression":
        return f"Relationship and unusual rows · {result.column}"
    return f"{result.test_name.replace('_', ' ').title()} · {result.column}"


def _image(fig):
    buffer = io.BytesIO()
    try:
        style_figure(fig).savefig(buffer, format="png", dpi=200, bbox_inches="tight")
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    finally:
        plt.close(fig)


def build_report(bundle, *, top_n=20):
    """Create portable report data; Findings alone supply written readings."""
    sections = []
    confidence = bundle.run_metadata.get("config", {}).get("benford_band", 0.95)
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
        if result.column in bundle.dataset_profile["column_profiles"]:
            profile = bundle.dataset_profile["column_profiles"][result.column]
            numbers.update(
                {
                    key: profile[key]
                    for key in (
                        "finite_count",
                        "positive_count",
                        "finite_min",
                        "finite_max",
                        "positive_min",
                        "positive_max",
                        "positive_orders_span",
                        "ecdf_95_band",
                    )
                    if key in profile
                }
            )
        charts = []
        if name.startswith("benford_"):
            numbers["reference_band"] = f"{confidence:.0%}"
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
                        _image(digit_figure(result, view, confidence=confidence)),
                        f"The {confidence:.0%} band shows the expected range for each digit. Orange bars are outside it. A narrower band marks more bars on the chart only.",
                    )
                )
            if f"{i}:mantissa" in bundle.figures:
                charts.append(
                    (
                        "mantissa",
                        _image(bundle.figures[f"{i}:mantissa"]()),
                        "Compare how often values occur across the log-scaled range with the flat reference line.",
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
                    "A wider space between the two lines means the data differ more from the chosen reference.",
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
                    "Look for points far from the main cluster or a changing spread across the chart.",
                )
            )
            if "groups" in result.details:
                charts.append(
                    (
                        "groups",
                        _image(plot_group_residuals(result)),
                        "Compare how far each group's values sit above or below the model's predictions.",
                    )
                )
        section_findings = [
            f
            for f in bundle.findings
            if f.section == name and f.column == result.column
        ]
        sections.append(
            {
                "title": _title(result),
                "numbers": numbers,
                "number_rows": [
                    (
                        NUMBER_LABELS.get(key, key.replace("_", " ").title()),
                        value,
                        metric_explanation(result, key),
                    )
                    for key, value in numbers.items()
                    if key not in ("bins", "observed", "expected", "z_scores")
                ],
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
                "number_rows": [
                    (
                        NUMBER_LABELS["index"],
                        ml.index,
                        metric_explanation(None, "index"),
                    ),
                    (
                        "Share of records above the review threshold",
                        f"{ml.components['review_share']:.1%}",
                        metric_explanation(None, "review_share"),
                    ),
                    (
                        "Combined size of differences (0–1)",
                        f"{ml.components['normalized_effect']:.2f}",
                        metric_explanation(None, "normalized_effect"),
                    ),
                    (
                        "Combined reference-test signal (0–1)",
                        f"{ml.components['adjusted_significance']:.2f}",
                        metric_explanation(None, "adjusted_significance"),
                    ),
                    (
                        NUMBER_LABELS["threshold"],
                        ml.threshold,
                        metric_explanation(None, "threshold"),
                    ),
                    (
                        NUMBER_LABELS["reviewed_records"],
                        len(ml.flagged_indices),
                        metric_explanation(None, "reviewed_records"),
                    ),
                ],
                "charts": [
                    (
                        "scores",
                        _image(score_figure(ml.scores, ml.threshold)),
                        "The farther right a score sits, the more unusual that record is compared with others here.",
                    ),
                    (
                        "top",
                        _image(top_records_figure(ml.scores, ml.contributions, top_n)),
                        "These bars show the records most worth checking, with a measurement that stood out for each.",
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
        "font_css": font_face_css(),
        "metadata": bundle.run_metadata,
        "profile": bundle.dataset_profile,
        "cover": cover,
        "sections": sections,
        "summary": summary,
        "flagged_rows": sorted(
            {p for r in bundle.results for p in r.flagged_indices}
            | (set(ml.flagged_indices) if ml else set())
        )[:top_n],
        "index": ml.index if ml and ml.scores else None,
        "components": ml.components if ml and ml.scores else {},
        "scoring_warning": (
            ml.warning if ml else bundle.run_metadata.get("scoring_warning")
        ),
    }
