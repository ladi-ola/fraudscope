"""Single UI-independent entry point for analysis of in-memory frames."""

import hashlib
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

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
        columns[str(name)] = {
            "dtype": str(df[name].dtype),
            "missing": int(df[name].isna().sum()),
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
            metadata["package_versions"]["analysis package" if package == "fraudscope" else package] = version(package)
        except PackageNotFoundError:
            metadata["package_versions"]["analysis package" if package == "fraudscope" else package] = "not installed"
    bundle = AnalysisBundle(results, profile, metadata)
    if config.ml_enabled:
        from .ml.anomaly import score_records

        bundle.ml_output = score_records(df, config, results)
    from .findings import FindingsEngine

    bundle.findings = FindingsEngine().build(results, bundle.ml_output)
    return bundle
