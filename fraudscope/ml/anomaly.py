"""Deterministic Isolation Forest rankings and a transparent dataset index."""

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from .explain import top_contributors
from .features import make_features

MIN_ROWS = 30
THRESHOLD = 0.9


@dataclass
class AnomalyOutput:
    scores: list[float] = field(default_factory=list)
    baseline: list[float] = field(default_factory=list)
    flagged_indices: list[int] = field(default_factory=list)
    contributions: dict[int, list[str]] = field(default_factory=dict)
    index: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    warning: str | None = None
    threshold: float = THRESHOLD


def anomaly_index(results, scores, threshold=THRESHOLD):
    """100*(.2*review_share + .5*mean normalized effect + .3*mean significance).

    Effects: Benford MAD / .015 (first), .012 (second), .0022 (first-two),
    last-two MAD / .01; KS D / .2; Kuiper V / .3. All clipped to [0,1].
    Regression F-test p measures association, not anomalies: excluded.
    Significance: max(0, 1 - adjusted_p/.05); only eligible distribution tests.
    These are descriptive design scales, not universal critical values.
    """
    scales = {
        "benford_first": 0.015,
        "benford_second": 0.012,
        "benford_first_two": 0.0022,
        "benford_last_two": 0.01,
        "ks_one_sample": 0.2,
        "kuiper_one_sample": 0.3,
    }
    eligible = [
        r
        for r in results
        if r.test_name in scales and r.effect_size is not None and r.p_value is not None
    ]
    effect = (
        float(
            np.mean(
                [min(1, max(0, r.effect_size / scales[r.test_name])) for r in eligible]
            )
        )
        if eligible
        else 0.0
    )
    significance = (
        float(
            np.mean(
                [
                    max(0, 1 - r.details.get("p_adjusted", r.p_value) / 0.05)
                    for r in eligible
                ]
            )
        )
        if eligible
        else 0.0
    )
    share = float(np.mean(np.asarray(scores) >= threshold)) if len(scores) else 0.0
    components = {
        "review_share": share,
        "normalized_effect": effect,
        "adjusted_significance": significance,
    }
    return round(100 * (0.2 * share + 0.5 * effect + 0.3 * significance), 2), components


def score_records(df, config, results):
    """Score at least 30 rows, with median imputation and missing indicators."""
    n = len(df)
    baseline = np.zeros(n)
    for result in results:
        if result.p_value is None:
            continue
        positions = np.asarray(result.flagged_indices, dtype=int)
        baseline[positions[(positions >= 0) & (positions < n)]] += 1
    if n < MIN_ROWS:
        return AnomalyOutput(
            baseline=baseline.tolist(),
            warning=f"Anomaly scoring requires at least {MIN_ROWS} rows.",
        )
    features = make_features(df, config, results)
    if features.empty or features.notna().sum().sum() == 0:
        return AnomalyOutput(
            baseline=baseline.tolist(),
            warning="No usable numeric features for anomaly scoring.",
        )
    features = features.loc[:, features.notna().any()]
    missing = features.isna().astype(float).add_suffix(":missing")
    missing = missing.loc[:, missing.nunique() > 1]
    imputed = SimpleImputer(strategy="median").fit_transform(features)
    matrix = np.column_stack([imputed, missing.to_numpy()])
    names = list(features.columns) + list(missing.columns)
    variable = np.ptp(matrix, axis=0) > 1e-12
    matrix, names = matrix[:, variable], [
        key for key, keep in zip(names, variable) if keep
    ]
    if not names:
        return AnomalyOutput(
            baseline=baseline.tolist(), warning="All model features are constant."
        )
    matrix = RobustScaler().fit_transform(matrix)
    forest = IsolationForest(
        random_state=config.random_state if config.random_state is not None else 0,
        contamination="auto",
        n_estimators=100,
    )
    raw = -forest.fit(matrix).score_samples(matrix)
    order = np.argsort(raw, kind="stable")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n) / max(1, n - 1)
    flagged = np.flatnonzero(ranks >= THRESHOLD).tolist()
    index, components = anomaly_index(results, ranks)
    return AnomalyOutput(
        ranks.tolist(),
        baseline.tolist(),
        flagged,
        top_contributors(matrix, names, flagged),
        index,
        components,
    )
