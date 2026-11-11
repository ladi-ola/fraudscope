"""Linear model diagnostics and robust coefficient comparison."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from benford.digits import expected_probs
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

from .results import TestResult


def regression_test(
    df: pd.DataFrame,
    y: str,
    x: list[str],
    *,
    group: str | None = None,
    order: str | None = None,
    strict=False,
) -> TestResult:
    """OLS, Huber RLM and row influence; p-value is the OLS F-test p-value.

    Standard cutoffs: |externally studentized residual|>3, Cook's D>4/n,
    leverage>2k/n (k is the number of fitted parameters incl. intercept).
    The F-test describes explanatory association, not anomalous records.
    """
    if not x or y in x:
        raise ValueError("Specify predictors distinct from the response.")
    columns = list(
        dict.fromkeys([y, *x, *([group] if group else []), *([order] if order else [])])
    )
    frame = df[columns].copy()
    for col in [y, *x]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame.loc[~np.isfinite(frame[col]), col] = np.nan
    frame["_position"] = np.arange(len(frame))
    frame = frame.dropna(subset=[y, *x])
    warnings = []
    if len(frame) < max(30, len(x) + 3):
        warnings.append("Small sample relative to model complexity.")
    if strict and warnings:
        raise ValueError(" ".join(warnings))
    if len(frame) <= len(x) + 2:
        raise ValueError("Insufficient complete rows for regression diagnostics.")
    exog = sm.add_constant(frame[x].astype(float), has_constant="add")
    if np.linalg.matrix_rank(exog) < exog.shape[1]:
        raise ValueError("Predictors are collinear or constant.")
    endog = frame[y].astype(float)
    ols = sm.OLS(endog, exog).fit()
    robust = sm.RLM(endog, exog, M=sm.robust.norms.HuberT()).fit()
    influence = ols.get_influence()
    student = np.asarray(influence.resid_studentized_external)
    leverage = np.asarray(influence.hat_matrix_diag)
    cooks = np.asarray(influence.cooks_distance[0])
    dffits = np.asarray(influence.dffits[0])
    n, k = len(frame), exog.shape[1]
    positions = frame["_position"].astype(int).to_numpy()
    severe = (np.abs(student) > 3) | (cooks > 4 / n) | (leverage > 2 * k / n)
    score = np.maximum.reduce(
        [np.abs(student) / 3, cooks / (4 / n), leverage / (2 * k / n)]
    )
    flagged = positions[np.flatnonzero(severe)[np.argsort(-score[severe])]].tolist()
    scale = np.maximum(
        np.abs(ols.params.to_numpy()), float(ols.resid.std()) * 0.1 + 1e-10
    )
    if np.any(np.abs(ols.params.to_numpy() - robust.params.to_numpy()) / scale > 0.25):
        warnings.append(
            "OLS and Huber coefficients differ materially (>25% relative scale)."
        )
    bp = het_breuschpagan(ols.resid, exog)
    if n <= 5000:
        normal_stat, normal_p = stats.shapiro(ols.resid)
        normal_name = "shapiro_wilk"
    else:
        normal_stat, normal_p, _, _ = sm.stats.jarque_bera(ols.resid)
        normal_name = "jarque_bera"
    if bp[1] < 0.05:
        warnings.append("Breusch-Pagan suggests nonconstant residual variance.")
    if normal_p < 0.05:
        warnings.append("Residual normality check suggests a deviation.")
    details = {
        "ols_coefficients": ols.params.to_dict(),
        "robust_coefficients": robust.params.to_dict(),
        "r_squared": float(ols.rsquared),
        "adj_r_squared": float(ols.rsquared_adj),
        "f_statistic": float(ols.fvalue),
        "f_p_value": float(ols.f_pvalue),
        "residual_standard_error": float(np.sqrt(ols.mse_resid)),
        "fitted": ols.fittedvalues.tolist(),
        "residuals": ols.resid.tolist(),
        "studentized_residuals": student.tolist(),
        "leverage": leverage.tolist(),
        "cooks_distance": cooks.tolist(),
        "dffits": dffits.tolist(),
        "row_positions": positions.tolist(),
        "breusch_pagan": {"statistic": float(bp[0]), "p_value": float(bp[1])},
        "normality": {
            "test": normal_name,
            "statistic": float(normal_stat),
            "p_value": float(normal_p),
        },
    }
    if order:
        ordered_residuals = frame.assign(_resid=ols.resid).sort_values(order)["_resid"]
        dw = float(durbin_watson(ordered_residuals))
        details["durbin_watson"] = dw
        if dw < 1.5 or dw > 2.5:
            warnings.append(
                "Durbin-Watson suggests serial dependence in the specified order."
            )
    if group:
        groups = (
            frame.assign(_resid=ols.resid)
            .dropna(subset=[group])
            .groupby(group)["_resid"]
        )
        group_values = [v.to_numpy() for _, v in groups]
        details["groups"] = {
            str(key): {
                "mean_residual": float(v.mean()),
                "residuals": v.tolist(),
                "n": len(v),
            }
            for key, v in groups
        }
        if len(group_values) >= 2 and all(len(v) > 1 for v in group_values):
            stat, p = stats.f_oneway(*group_values)
            details["group_anova"] = {"statistic": float(stat), "p_value": float(p)}
            if p < 0.05:
                warnings.append(
                    "Group residual means differ; composition or specification may explain this."
                )
        rse = np.sqrt(ols.mse_resid)
        details["flagged_groups"] = [
            name
            for name, item in details["groups"].items()
            if abs(item["mean_residual"]) > 2 * rse / np.sqrt(item["n"])
        ]
    return TestResult(
        "regression",
        y,
        float(ols.fvalue),
        float(ols.f_pvalue),
        "asymptotic",
        float(ols.rsquared),
        n,
        details,
        flagged,
        [
            "Association is not causation.",
            "Outliers can be legitimate.",
            "Model misspecification can produce misleading flags.",
        ],
        warnings,
    )


def digit_frequency_regression(observed, mode="first") -> TestResult:
    """OLS of observed digit proportions on Benford expected proportions."""
    bins, expected = expected_probs(mode)
    observed = np.asarray(observed, dtype=float)
    if observed.shape != expected.shape:
        raise ValueError("Observed proportions must match the reference bins.")
    fit = sm.OLS(observed, sm.add_constant(expected)).fit()
    return TestResult(
        "digit_frequency_regression",
        None,
        float(fit.rsquared),
        float(fit.f_pvalue),
        "asymptotic",
        float(fit.rsquared),
        len(bins),
        {
            "slope": float(fit.params[1]),
            "intercept": float(fit.params[0]),
            "r_squared": float(fit.rsquared),
        },
        [],
        ["Digit frequency association does not establish record-level anomalies."],
        [],
    )
