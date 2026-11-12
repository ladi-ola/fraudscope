"""Fault-isolated execution and false-discovery-rate adjustment."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from .benford import benford_test
from .ks import ks_one_sample
from .kuiper import kuiper_one_sample
from .regression import regression_test
from .results import TestResult


@dataclass
class RegressionSpec:
    y: str
    x: list[str]
    group: str | None = None
    order: str | None = None


@dataclass
class Config:
    columns: list[str]
    benford_modes: list[str] = field(default_factory=lambda: ["first"])
    references: dict = field(default_factory=dict)
    regression: RegressionSpec | None = None
    random_state: int | None = None
    strict: bool = False
    n_sim: int = 10_000
    top_k: int = 3
    decimals: int = 2
    ml_enabled: bool = True
    benford_band: float = 0.95


def run_all(df: pd.DataFrame, config: Config) -> list[TestResult]:
    """Run configured tests independently; BH-adjust all valid p-values."""
    results = []

    def attempt(name, column, operation):
        try:
            results.append(operation())
        except Exception as exc:  # noqa: BLE001 - isolate each user-configured test
            results.append(
                TestResult(
                    name,
                    column,
                    0.0,
                    None,
                    "none",
                    None,
                    0,
                    warnings=[f"Test could not run: {type(exc).__name__}: {exc}"],
                    caveats=["No conclusion can be drawn from a failed analysis."],
                )
            )

    for col in config.columns:
        for mode in config.benford_modes:
            attempt(
                "benford_" + mode,
                col,
                lambda col=col, mode=mode: benford_test(
                    df[col],
                    mode,
                    column=col,
                    top_k=config.top_k,
                    decimals=config.decimals,
                    strict=config.strict,
                ),
            )
        if col in config.references:
            reference = config.references[col]
            attempt(
                "ks_one_sample",
                col,
                lambda col=col, ref=reference: ks_one_sample(
                    df[col],
                    ref,
                    n_sim=config.n_sim,
                    random_state=config.random_state,
                    strict=config.strict,
                    column=col,
                ),
            )
            attempt(
                "kuiper_one_sample",
                col,
                lambda col=col, ref=reference: kuiper_one_sample(
                    df[col],
                    ref,
                    n_sim=config.n_sim,
                    random_state=config.random_state,
                    strict=config.strict,
                    column=col,
                ),
            )
    if config.regression:
        spec = config.regression
        attempt(
            "regression",
            spec.y,
            lambda: regression_test(
                df,
                spec.y,
                spec.x,
                group=spec.group,
                order=spec.order,
                strict=config.strict,
            ),
        )
    valid = [r for r in results if r.p_value is not None and np.isfinite(r.p_value)]
    if valid:
        adjusted = multipletests([r.p_value for r in valid], method="fdr_bh")[1]
        for result, p in zip(valid, adjusted):
            result.details["p_adjusted"] = float(p)
    return results
