"""Vectorized record features aligned to zero-based input positions."""

import numpy as np
import pandas as pd
from benford.digits import extract_digits

from fraudscope._ecdf import reference_data


def _numeric(series):
    return pd.to_numeric(series.reset_index(drop=True), errors="coerce").to_numpy(
        dtype=float
    )


def make_features(df, config, results):
    """Build model-derived and generic features; missing rows remain NaN."""
    n = len(df)
    features = pd.DataFrame(index=pd.RangeIndex(n))
    for col in config.columns:
        if col not in df:
            continue
        values = _numeric(df[col])
        finite = np.isfinite(values)
        s = pd.Series(values)
        median = np.nanmedian(values[finite]) if finite.any() else 0
        mad = np.nanmedian(np.abs(values[finite] - median)) if finite.any() else 0
        features[f"{col}:robust_z"] = np.abs(values - median) / max(1.4826 * mad, 1e-8)
        logged = np.where(
            finite & (values > 0), np.log10(np.maximum(values, 1e-300)), np.nan
        )
        lm = np.nanmedian(logged) if np.isfinite(logged).any() else 0
        ld = np.nanmedian(np.abs(logged - lm)) if np.isfinite(logged).any() else 0
        features[f"{col}:log_robust_z"] = np.abs(logged - lm) / max(1.4826 * ld, 1e-8)
        for base in (10, 100, 1000):
            features[f"{col}:multiple_{base}"] = np.where(
                finite, np.isclose(values % base, 0), np.nan
            )
        counts = s[finite].value_counts(normalize=True)
        features[f"{col}:repeat_frequency"] = s.map(counts).to_numpy()
        for result in results:
            if result.column != col or result.n == 0:
                continue
            prefix = f"{col}:{result.test_name}"
            if result.test_name.startswith("benford_"):
                mode = result.details["mode"]
                digits = extract_digits(s, mode, config.decimals)
                probability = dict(
                    zip(result.details["bins"], result.details["expected"])
                )
                features[f"{prefix}:surprisal"] = (
                    -np.log(digits.map(probability)).reindex(range(n)).to_numpy()
                )
                deviation = dict(
                    zip(result.details["bins"], np.abs(result.details["deviation"]))
                )
                features[f"{prefix}:bin_deviation"] = (
                    digits.map(deviation).reindex(range(n)).to_numpy()
                )
                features[f"{prefix}:flag"] = (
                    pd.Series(
                        digits.isin(result.details["flagged_bins"]).astype(float),
                        index=digits.index,
                    )
                    .reindex(range(n))
                    .to_numpy()
                )
            elif result.test_name in ("ks_one_sample", "kuiper_one_sample"):
                reference = config.references.get(col)
                if reference is None:
                    continue
                try:
                    eligible, cdf, _, _ = reference_data(
                        pd.Series(values[finite], index=np.flatnonzero(finite)),
                        reference,
                    )
                    positions = eligible.index.to_numpy(dtype=int)
                    ordered = np.argsort(eligible.to_numpy(), kind="stable")
                    ranks = np.empty(len(ordered))
                    ranks[ordered] = (np.arange(len(ordered)) + 0.5) / len(ordered)
                    deviation = np.full(n, np.nan)
                    probability = np.asarray(cdf(eligible.to_numpy()), dtype=float)
                    deviation[positions] = np.abs(ranks - probability)
                    tail = np.full(n, np.nan)
                    tail[positions] = (probability < 0.05) | (probability > 0.95)
                    features[f"{prefix}:ecdf_gap"] = deviation
                    features[f"{prefix}:tail"] = tail
                except (ValueError, TypeError):
                    continue
    for result in results:
        if result.test_name != "regression" or result.n == 0:
            continue
        d = result.details
        pos = np.asarray(d["row_positions"], dtype=int)
        for key in ("studentized_residuals", "cooks_distance", "leverage"):
            feature = np.full(n, np.nan)
            feature[pos] = np.abs(d[key])
            features[f"regression:{key}"] = feature
        residual = np.asarray(d["residuals"])
        robust = np.asarray(d["ols_coefficients"].get("const", 0)) - np.asarray(
            d["robust_coefficients"].get("const", 0)
        )
        if config.regression:
            for col in config.regression.x:
                robust += np.asarray(df[col].iloc[pos], dtype=float) * (
                    d["ols_coefficients"][col] - d["robust_coefficients"][col]
                )
        gap = np.full(n, np.nan)
        gap[pos] = np.abs(robust)
        features["regression:ols_robust_gap"] = gap
        if config.regression and config.regression.group and "groups" in d:
            group = df[config.regression.group].iloc[pos].astype(str)
            means = group.map(
                {k: v["mean_residual"] for k, v in d["groups"].items()}
            ).to_numpy(dtype=float)
            relative = np.full(n, np.nan)
            relative[pos] = np.abs(residual - means)
            features["regression:group_relative_residual"] = relative
    return features.replace([np.inf, -np.inf], np.nan)
