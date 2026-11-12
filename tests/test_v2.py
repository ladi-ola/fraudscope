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
from sklearn.metrics import roc_auc_score

from fraudscope.findings import FindingsEngine, severity
from fraudscope.report.export import findings_json, html_report, pdf_report
from fraudscope.report.figures import digit_figure
from fraudscope.runner import Config
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
    assert tiny.ml_output.warning
    constant = run_analysis(
        pd.DataFrame({"amount": [10.0] * 35}), Config(["amount"], benford_modes=[])
    )
    assert constant.ml_output.warning


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
    assert b"%PDF" in pdf_report(bundle)[:10]
    assert json.loads(findings_json(bundle))["schema_version"] == "1.0"
    for text in (html, findings_json(bundle)):
        assert not any(
            word in text.lower() for word in ("fraudulent", "rigged", "fraud")
        )


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
    assert run_analysis(clean, config).ml_output.index < run_analysis(data, config).ml_output.index
