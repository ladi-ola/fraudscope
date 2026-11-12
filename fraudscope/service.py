"""Single UI-independent entry point for analysis of in-memory frames."""

import hashlib
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pandas as pd

from .applicability import benford_applicable, continuous_applicable
from .bundle import AnalysisBundle
from .runner import Config, run_all


def profile_dataset(df: pd.DataFrame) -> dict:
    """Profile columns without retaining or logging any uploaded values."""
    columns = {}
    for name in df:
        benford_ok, benford_reasons = benford_applicable(df[name])
        continuous_ok, continuous_reasons = continuous_applicable(df[name])
        numeric = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
        finite = numeric[np.isfinite(numeric)]
        positive = finite[finite > 0]
        columns[str(name)] = {
            "dtype": str(df[name].dtype),
            "missing": int(df[name].isna().sum()),
            "finite_count": len(finite),
            "positive_count": len(positive),
            "finite_min": float(finite.min()) if len(finite) else None,
            "finite_max": float(finite.max()) if len(finite) else None,
            "positive_min": float(positive.min()) if len(positive) else None,
            "positive_max": float(positive.max()) if len(positive) else None,
            "positive_orders_span": (
                float(np.log10(positive.max()) - np.log10(positive.min()))
                if len(positive)
                else None
            ),
            "ecdf_95_band": (
                float(min(1, np.sqrt(np.log(40) / (2 * len(finite)))))
                if len(finite)
                else None
            ),
            "benford": "ok" if benford_ok else "caution",
            "benford_reasons": benford_reasons,
            "continuous": "ok" if continuous_ok else "caution",
            "continuous_reasons": continuous_reasons,
        }
    return {"rows": len(df), "columns": len(df.columns), "column_profiles": columns}


def run_analysis(
    df: pd.DataFrame,
    config: Config,
    *,
    data_hash: str | None = None,
    dataset_name: str = "dataset",
) -> AnalysisBundle:
    """Run independently isolated tests; later layers consume this bundle."""
    start = time.perf_counter()
    profile = profile_dataset(df)
    results = run_all(df, config)
    if data_hash is None:
        # Deterministic content digest, including row order and missing markers.
        values = pd.util.hash_pandas_object(
            df.reset_index(drop=True), index=False
        ).values
        data_hash = hashlib.sha256(values.tobytes()).hexdigest()
    metadata = {
        "dataset_name": dataset_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sha256": data_hash,
        "config": json.loads(json.dumps(asdict(config), default=str)),
        "runtime_seconds": round(time.perf_counter() - start, 4),
    }
    metadata["package_versions"] = {}
    for package in ("fraudscope", "numpy", "pandas", "scipy", "scikit-learn"):
        try:
            metadata["package_versions"][
                "analysis package" if package == "fraudscope" else package
            ] = version(package)
        except PackageNotFoundError:
            metadata["package_versions"][
                "analysis package" if package == "fraudscope" else package
            ] = "not installed"
    bundle = AnalysisBundle(results, profile, metadata)
    if config.ml_enabled:
        try:
            from .ml.anomaly import score_records
        except ModuleNotFoundError as exc:
            if exc.name != "sklearn":
                raise
            metadata["scoring_warning"] = (
                "Install the app extra to enable anomaly scoring."
            )
        else:
            bundle.ml_output = score_records(df, config, results)
    from .findings import FindingsEngine

    bundle.findings = FindingsEngine().build(results, bundle.ml_output)
    # Lazily construct figures so callers control display and closing.
    from .report.figures import (
        digit_figure,
        mantissa_figure,
        plot_ecdf_vs_reference,
        plot_residuals,
        score_figure,
    )

    for i, result in enumerate(results):
        if result.p_value is None:
            continue
        if result.test_name.startswith("benford_"):
            bundle.figures[f"{i}:primary"] = partial(digit_figure, result)
            bundle.figures[f"{i}:log"] = partial(digit_figure, result, "log")
            bundle.figures[f"{i}:deviation"] = partial(
                digit_figure, result, "deviation"
            )
            if result.column in df:
                values = pd.to_numeric(df[result.column], errors="coerce").to_numpy(
                    dtype=float
                )
                bundle.figures[f"{i}:mantissa"] = partial(mantissa_figure, values)
        elif result.test_name in ("ks_one_sample", "kuiper_one_sample"):
            bundle.figures[f"{i}:ecdf"] = partial(plot_ecdf_vs_reference, result)
        elif result.test_name == "regression":
            bundle.figures[f"{i}:residuals"] = partial(plot_residuals, result)
    if bundle.ml_output and bundle.ml_output.scores:
        bundle.figures["ml:scores"] = partial(score_figure, bundle.ml_output.scores)
    return bundle
