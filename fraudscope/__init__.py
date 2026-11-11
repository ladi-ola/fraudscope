"""Statistical anomaly detection; results are not proof of wrongdoing."""

from .results import TestResult
from .runner import Config, RegressionSpec, run_all

__all__ = ["TestResult", "Config", "RegressionSpec", "run_all"]
