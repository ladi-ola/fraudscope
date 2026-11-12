"""UI-independent output of a complete analysis run."""

from dataclasses import dataclass, field
from typing import Any

from .results import TestResult


@dataclass
class AnalysisBundle:
    """Results and optional derived layers, keyed by positional input rows."""

    results: list[TestResult]
    dataset_profile: dict[str, Any]
    run_metadata: dict[str, Any]
    ml_output: Any = None
    findings: list[Any] = field(default_factory=list)
    figures: dict[str, Any] = field(default_factory=dict)
