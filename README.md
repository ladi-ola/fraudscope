# fraudscope

Python 3.10+ statistical anomaly detection for tabular data. **An anomaly is
not proof of wrongdoing.** No learning models, hosted services or API keys.

## Install

```bash
python -m pip install -e '.[dev]'
python -m pytest -q tests benford_py/tests
```

One editable install exposes both `fraudscope` and the upstream `benford`
package. The upstream BSD-3-Clause license and attribution remain in
`benford_py/LICENSE.txt`. For the upstream baseline and additive modifications
see `CHANGES.md`. Run the four examples in `notebooks/` with Jupyter.

## Quickstart

```python
import numpy as np
import pandas as pd
from scipy.stats import norm
from fraudscope.ingest import load
from fraudscope.runner import Config, RegressionSpec, run_all
from fraudscope.plots import plot_summary_dashboard

df = pd.DataFrame({"amount": 10 ** np.random.default_rng(2).uniform(1, 5, 2000),
                   "sequence": np.arange(2000)})
results = run_all(df, Config(columns=["amount"],
                             benford_modes=["first", "second", "first_two", "last_two"],
                             references={"amount": "benford"}, random_state=2))
print([r.to_dict() for r in results])
plot_summary_dashboard(results, "overview.png")
# Or: df = load("data.csv", columns=["amount", "sequence"])
```

`load` accepts CSV and XLSX, trims whitespace, converts fully numeric string
columns, retains all rows, and stores cleaning counts in `df.attrs['cleaning']`.
Record flags are **zero-based input row positions**, not DataFrame index labels.

## What the tests mean

- **Benford**: leading digits for naturally generated, unbounded magnitudes;
  second, first-two and last-two (uniform cents by default) are supported.
  MAD is the main descriptive measure; chi-square becomes sensitive to tiny
  differences at large n. Last-two has no published conformity thresholds.
  Bounded values, identifiers, pricing and rounding may depart innocently.
  Multiplying a genuinely Benford-distributed subset by a constant preserves
  its leading-digit law; such scaling is not reliably detectable by digit tests
  alone. Compare against a known reference or groups to inspect range changes.
- **KS**: largest one- or two-sample ECDF gap; most appropriate for continuous
  references chosen independently of the sample. Discrete references must be
  supplied as `(sorted_bins, probabilities)` and use seeded Monte Carlo p-values.
- **Kuiper**: sum of positive and negative ECDF gaps; suited to circular data
  (time-of-day) and shift/tail sensitivity. Its asymptotic p-value is approximate;
  use Monte Carlo/permutation for small or discrete samples.
- **Regression**: OLS and Huber estimates, influence measures and residual
  diagnostics; a flagged observation can be genuine and assumptions or omitted
  variables can explain patterns. Group comparisons are exploratory.

For `reference="benford"`, **positive values** are transformed to fractional
`log10(x)` mantissas and compared to a uniform [0,1) CDF. This is not the
discrete digit reference. `strict=True` raises when applicability heuristics
fail in direct tests; the runner catches and reports failures independently.
Every result has innocent-explanation `caveats`, test `warnings`, and optional
BH-adjusted `details['p_adjusted']` across the run. Row flags locate bins or
diagnostics to inspect; they do not attribute intent to an individual record.
