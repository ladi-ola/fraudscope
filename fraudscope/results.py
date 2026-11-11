"""Common, JSON-compatible result contract."""

from dataclasses import asdict, dataclass, field
import json

import numpy as np


def _native(value):
    if isinstance(value, dict):
        return {str(k): _native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_native(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass
class TestResult:
    test_name: str
    column: str | None
    statistic: float | dict[str, float]
    p_value: float | None
    p_value_method: str
    effect_size: float | None
    n: int
    details: dict = field(default_factory=dict)
    flagged_indices: list[int] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert numpy scalars, arrays, and integer-keyed bins to JSON values."""
        return _native(asdict(self))

    def to_json(self) -> str:
        """Serialize the complete result without a custom encoder."""
        return json.dumps(self.to_dict(), allow_nan=False)
