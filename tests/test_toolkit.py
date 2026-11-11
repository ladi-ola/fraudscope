"""Seeded, offline checks for every result family and reference calculations."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy import stats
import statsmodels.api as sm

from benford.digits import expected_probs, extract_digits
from fraudscope.applicability import benford_applicable, continuous_applicable
from fraudscope.benford import benford_test
from fraudscope.ingest import load
from fraudscope.ks import ks_one_sample, ks_two_sample
from fraudscope.kuiper import kuiper_one_sample, kuiper_two_sample
from fraudscope.plots import (plot_benford, plot_ecdf_vs_reference, plot_group_residuals,
                              plot_residuals, plot_summary_dashboard)
from fraudscope.regression import digit_frequency_regression, regression_test
from fraudscope.runner import Config, RegressionSpec, run_all


def generated(seed, n=6000, anomaly=None):
    """Reproducible wide-ranging magnitudes with optional digit anomalies."""
    rng = np.random.default_rng(seed)
    values = 10 ** rng.uniform(2, 6, n)
    if anomaly == "preference":
        values[:n // 3] = 5 * 10 ** rng.uniform(2, 5, n // 3).astype(int)
    elif anomaly == "rounding":
        values = np.round(values / 100) * 100
    elif anomaly == "uniform":
        values = rng.integers(1, 10, n) * 10 ** rng.integers(2, 6, n)
    elif anomaly == "duplicates":
        values[:n // 3] = 55500
    elif anomaly == "scaled":
        values[:n // 3] *= 5
    return pd.Series(values, name="amount")


def test_digit_helpers():
    assert np.isclose(expected_probs("last_two")[1].sum(), 1)
    assert extract_digits(pd.Series([.000512, 9.99, 1023, 0, -23, np.nan]), "first").tolist() == [5, 9, 1]
    assert extract_digits(pd.Series([1234, 1299, 0]), "last_two", decimals=0).tolist() == [34, 99]
    for mode in ("first", "second", "first_two", "last_two"):
        result = benford_test(generated(13), mode)
        assert result.n > 0 and result.caveats
        assert len(result.details["bins"]) == len(result.details["expected"])
        json.loads(result.to_json())


@pytest.mark.parametrize("seed", [1, 7, 19])
def test_benford_null(seed):
    result = benford_test(generated(seed), "first")
    assert result.details["conformity"] == "close"
    assert result.p_value > 0.01


@pytest.mark.parametrize("anomaly,mode", [("preference", "first"), ("rounding", "last_two"),
                                            ("uniform", "first"), ("duplicates", "first")])
def test_benford_injections(anomaly, mode):
    result = benford_test(generated(3, anomaly=anomaly), mode, decimals=0)
    assert result.p_value < 0.01
    assert result.flagged_indices


def test_applicability_and_edges():
    assert not benford_applicable([0, -1, np.nan, 1])[0]
    assert not continuous_applicable([1] * 40)[0]
    with pytest.raises(ValueError):
        benford_test(pd.Series([0, -1, np.nan]), strict=True)
    result = benford_test(pd.Series([0, -1, np.nan]))
    assert result.n == 0 and result.p_value is None and result.warnings
    assert len(benford_test(pd.Series([555] * 301)).flagged_indices) == 301


def test_ks_reference_and_two_sample():
    rng = np.random.default_rng(24)
    a, b = rng.normal(size=300), rng.normal(loc=.4, size=180)
    one = ks_one_sample(pd.Series(a), stats.norm())
    two = ks_two_sample(a, b)
    assert one.statistic == pytest.approx(stats.ks_1samp(a, stats.norm().cdf).statistic)
    assert two.statistic == pytest.approx(stats.ks_2samp(a, b).statistic)
    assert one.details["D_plus"] >= 0 and one.details["D_minus"] >= 0
    assert 0 < ks_one_sample(generated(2), "benford").p_value < 1


def test_discrete_calibration_and_reproducibility():
    bins, probs = expected_probs("first")
    rng = np.random.default_rng(8)
    draw = rng.choice(bins, 100, p=probs)
    a = ks_one_sample(draw, (bins, probs), n_sim=250, random_state=17)
    b = kuiper_one_sample(draw, (bins, probs), n_sim=250, random_state=17)
    assert a.p_value_method == b.p_value_method == "monte_carlo"
    assert a.p_value == ks_one_sample(draw, (bins, probs), n_sim=250, random_state=17).p_value
    assert 0 <= a.p_value <= 1 and 0 <= b.p_value <= 1


def test_kuiper_against_reference_and_null():
    rng = np.random.default_rng(45)
    p = []
    for _ in range(40):
        sample = rng.uniform(size=150)
        result = kuiper_one_sample(sample, stats.uniform())
        ordered = np.sort(sample)
        reference = max(np.arange(1, 151) / 150 - ordered) + max(ordered - np.arange(150) / 150)
        assert result.statistic == pytest.approx(reference, abs=1e-12)
        p.append(result.p_value)
    assert .3 < np.mean(p) < .7
    a, b = rng.uniform(size=80), rng.uniform(size=120)
    assert 0 <= kuiper_two_sample(a, b, n_sim=100, random_state=3, monte_carlo=True).p_value <= 1


def test_kuiper_against_astropy():
    """Both statistics agree with astropy to 1e-12; p approximations differ."""
    astropy = pytest.importorskip("astropy.stats")
    rng = np.random.default_rng(71)
    for _ in range(5):
        a, b = rng.uniform(size=120), rng.uniform(size=85)
        assert kuiper_one_sample(a, stats.uniform()).statistic == pytest.approx(
            astropy.kuiper(a, stats.uniform.cdf)[0], abs=1e-12)
        assert kuiper_two_sample(a, b).statistic == pytest.approx(
            astropy.kuiper_two(a, b)[0], abs=1e-12)


def test_regression_stats_and_group():
    rng = np.random.default_rng(14)
    x = rng.normal(size=160)
    y = 2 + 3 * x + rng.normal(scale=.4, size=160)
    y[8] += 10
    frame = pd.DataFrame({"y": y, "x": x, "group": np.tile(["a", "b"], 80), "order": np.arange(160)})
    result = regression_test(frame, "y", ["x"], group="group", order="order")
    sm_fit = sm.OLS(y, sm.add_constant(x)).fit()
    assert result.details["r_squared"] == pytest.approx(sm_fit.rsquared)
    assert result.details["ols_coefficients"]["x"] == pytest.approx(sm_fit.params[1])
    assert 8 in result.flagged_indices and result.caveats
    assert "group_anova" in result.details and "durbin_watson" in result.details
    assert digit_frequency_regression(expected_probs("first")[1]).details["r_squared"] == pytest.approx(1)


def test_ingest_runner_plots(tmp_path):
    df = pd.DataFrame({" amount ": generated(2, 800).to_numpy(), "x": np.arange(800)})
    path = tmp_path / "input.csv"
    df.to_csv(path, index=False)
    loaded = load(path, ["amount", "x"])
    assert loaded.attrs["cleaning"]["rows"] == 800
    results = run_all(loaded, Config(["amount", "missing"], references={"amount": "benford"},
                                    regression=RegressionSpec("amount", ["x"])))
    assert len(results) == 5
    assert any(r.warnings and r.p_value is None for r in results)
    assert all(r.caveats for r in results)
    assert all("p_adjusted" in r.details for r in results if r.p_value is not None)
    for fig in (plot_benford(results[0], tmp_path / "digits.png"),
                plot_ecdf_vs_reference(results[1]), plot_residuals(results[-1]),
                plot_summary_dashboard(results)):
        assert fig.axes
        plt.close(fig)
    group_result = regression_test(loaded.assign(g=np.arange(800) % 2), "amount", ["x"], group="g")
    plt.close(plot_group_residuals(group_result))
    assert (tmp_path / "digits.png").exists()
