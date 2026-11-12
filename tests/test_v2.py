"""Offline contract checks for the service, scoring, findings, figures and exports."""

import json
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from reportlab.pdfbase import pdfmetrics
from sklearn.metrics import roc_auc_score

from app.theme import CSS
from fraudscope.applicability import benford_applicable, continuous_applicable
from fraudscope.findings import FindingsEngine, severity
from fraudscope.report.export import (
    _pdf_image_size,
    findings_json,
    html_report,
    pdf_report,
)
from fraudscope.report.figures import digit_figure
from fraudscope.report.typography import font_path
from fraudscope.runner import Config, RegressionSpec
from fraudscope.service import run_analysis


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(9)
    values = 10 ** rng.uniform(1, 6, 350)
    values[:40] = 55500
    return pd.DataFrame({"amount": values})


def test_service_graceful_and_deterministic(data):
    options = Config(
        ["amount", "missing"], references={"amount": "benford"}, random_state=7
    )
    a = run_analysis(data, options)
    b = run_analysis(data, options)
    assert len(a.results) == 4
    assert any(r.p_value is None and r.warnings for r in a.results)
    assert a.ml_output.scores == b.ml_output.scores
    assert a.findings == b.findings
    assert a.dataset_profile["rows"] == 350
    assert len(a.run_metadata["sha256"]) == 64
    assert pickle.loads(pickle.dumps(a)).figures


def test_tiny_missing_and_constant():
    tiny = run_analysis(pd.DataFrame({"amount": [1.0, None, 2.0]}), Config(["amount"]))
    assert len(tiny.ml_output.scores) == 3
    assert tiny.dataset_profile["column_profiles"]["amount"]["positive_count"] == 2
    constant = run_analysis(
        pd.DataFrame({"amount": [10.0] * 35}), Config(["amount"], benford_modes=[])
    )
    assert constant.ml_output.warning
    single = run_analysis(pd.DataFrame({"amount": [7.0]}), Config(["amount"]))
    assert "Overall anomaly index: Not available" in html_report(single)
    assert pdf_report(single).startswith(b"%PDF")


def test_no_fixed_minimum_and_reported_sample_scale():
    values = [1.0, 400.0]
    assert benford_applicable(values, strict=True)[0]
    assert continuous_applicable(values, strict=True)[0]
    bundle = run_analysis(
        pd.DataFrame({"amount": values}), Config(["amount"], ml_enabled=False)
    )
    profile = bundle.dataset_profile["column_profiles"]["amount"]
    assert profile["finite_count"] == profile["positive_count"] == 2
    assert profile["positive_orders_span"] == pytest.approx(np.log10(400))
    assert profile["ecdf_95_band"] == pytest.approx(np.sqrt(np.log(40) / 4))
    assert bundle.results[0].n == 2 and bundle.results[0].p_value is not None
    html = html_report(bundle)
    assert "2 usable numbers" in html
    assert "400.0" in html and "uncertainty" in html
    assert pdf_report(bundle).startswith(b"%PDF")


def test_report_tolerates_older_profile_shape():
    bundle = run_analysis(
        pd.DataFrame({"amount": [1.0, 400.0]}),
        Config(["amount"], ml_enabled=False),
    )
    column = bundle.dataset_profile["column_profiles"]["amount"]
    column.pop("finite_count")
    column.pop("positive_count")
    assert "NUMBERS" in html_report(bundle)


def test_small_regression_uses_model_degrees_of_freedom():
    rng = np.random.default_rng(4)
    x = np.arange(12, dtype=float)
    frame = pd.DataFrame({"x": x, "y": 2 + 3 * x + rng.normal(size=12)})
    bundle = run_analysis(
        frame,
        Config(
            ["y"],
            benford_modes=[],
            regression=RegressionSpec("y", ["x"]),
            strict=True,
            ml_enabled=False,
        ),
    )
    assert bundle.results[0].n == 12
    assert bundle.results[0].p_value is not None


def test_findings_thresholds(data):
    assert severity(0.05, 0.05, 0.1) == "notable"
    assert severity(0.1, 0.05, 0.1) == "strong"
    result = run_analysis(data, Config(["amount"], ml_enabled=False)).results[0]
    finding = FindingsEngine().build([result])[0]
    assert f"{result.details['mad']:.4f}" in finding.statement
    assert finding.caveat_ids


@pytest.mark.parametrize("mode", ["first", "second", "first_two", "last_two"])
def test_digit_figures(data, mode):
    result = run_analysis(
        data, Config(["amount"], benford_modes=[mode], ml_enabled=False)
    ).results[0]
    for view in ("primary", "log", "deviation"):
        fig = digit_figure(result, view)
        assert isinstance(fig, Figure)
        assert fig.get_size_inches()[0] == pytest.approx(12)
        assert fig.get_size_inches()[1] == pytest.approx(6.5)
        assert fig.axes[0].title.get_fontproperties().get_file() == str(font_path(True))
        if view == "primary":
            assert fig.axes[0].patches and fig.axes[0].lines
        if view == "log":
            assert fig.axes[0].get_yscale() == "log"
        plt.close(fig)


def test_exports_and_wording(data):
    bundle = run_analysis(
        data, Config(["amount"], references={"amount": "benford"}, random_state=7)
    )
    html = html_report(bundle)
    for marker in ("NUMBERS", "CHART", "STATISTICAL READING", "data:image/png;base64"):
        assert marker in html
    assert "What we measured" in html and "How to interpret this report" in html
    assert "Usable numeric values in this column" in html
    assert "font-family:'Source Code Pro',sans-serif" in html
    assert "font-family:'Source Code Pro',sans-serif" in CSS
    assert b"%PDF" in pdf_report(bundle)[:10]
    assert "SourceCodePro" in pdfmetrics.getRegisteredFontNames()
    assert json.loads(findings_json(bundle))["schema_version"] == "1.0"
    for text in (html, findings_json(bundle)):
        assert not any(
            word in text.lower() for word in ("fraudulent", "rigged", "fraud")
        )


def test_tall_dashboard_fits_pdf_page():
    width, height = _pdf_image_size(900, 2000)
    assert height <= 650 and width <= 480
    assert width / height == pytest.approx(900 / 2000)


def test_ml_ablation_metrics(data, capsys):
    labels = np.zeros(len(data), dtype=int)
    labels[:40] = 1
    bundle = run_analysis(data, Config(["amount"], random_state=3))
    forest = roc_auc_score(labels, bundle.ml_output.scores)
    baseline = roc_auc_score(labels, bundle.ml_output.baseline)
    k = 40
    precision = lambda scores: labels[np.argsort(-np.asarray(scores))[:k]].mean()
    print(
        {
            "forest_auc": forest,
            "baseline_auc": baseline,
            "forest_precision_at_k": precision(bundle.ml_output.scores),
            "baseline_precision_at_k": precision(bundle.ml_output.baseline),
        }
    )
    assert "forest_auc" in capsys.readouterr().out


def test_index_clean_vs_digit_preference(data):
    rng = np.random.default_rng(9)
    clean = pd.DataFrame({"amount": 10 ** rng.uniform(1, 6, len(data))})
    config = Config(["amount"], random_state=3)
    assert (
        run_analysis(clean, config).ml_output.index
        < run_analysis(data, config).ml_output.index
    )
