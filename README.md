# fraudscope

Python 3.10+ statistical anomaly detection for tabular data. **An anomaly is
not proof of wrongdoing.** Optional unsupervised anomaly scoring; no hosted services or API keys.

## Install

```bash
python -m pip install -e '.[dev]'
python -m pytest -q tests benford_py/tests
```

## Web app

```bash
python -m pip install -e '.[app]'
streamlit run app/main.py
```

Upload a CSV/XLSX file or choose either built-in sample. Select columns and
digit modes, optionally compare to Benford log-mantissas, then run the analysis.
HTML, PDF and schema-versioned `findings.json` are available from the Report tab.
**Screenshots:** placeholder (capture the app's Overview and Report tabs).

Architecture:

```text
CSV/XLSX in memory → app/main.py → service.run_analysis → AnalysisBundle
                                             ├─ v1 independent tests
                                             ├─ optional Isolation Forest ranking
                                             └─ deterministic Findings
                   shared bundle → app tabs / HTML report / ReportLab PDF
```

The app does not upload data to any external service or persist uploaded data
to disk. Its default limit is 50 MB and 100,000 rows. Exports are assembled in
memory. The core v1 package remains usable without Streamlit installed.

Isolation Forest ranks records *relative to the current dataset*; its score and
the 0–100 descriptive index are **not probabilities or evidence of intent**.
Missing features receive median imputation and missing indicators; under 30
rows scoring is disabled. The index is 100 × (0.2 × review share + 0.5 × mean
normalized effect + 0.3 × mean adjusted significance), with components shown
in exports. These scales are design choices, not universal statistical cutoffs.
Effects are clipped at MAD .015/.012/.0022/.01 for first/second/first-two/
last-two digits, KS D .2, and Kuiper V .3. Significance is
`max(0, 1 - adjusted_p / .05)` for eligible distribution tests. The .2 and
.3 scales are pragmatic, not published KS/Kuiper critical values; regression
F-test significance is excluded because it measures association. Findings use
Nigrini MAD guidance for the first three digit modes; last-two and diagnostic
flag-rate severity bands are descriptive, not validated universal thresholds.
`notebooks/05_ml_validation.ipynb` computes ROC-AUC and precision@k against
the tests-only baseline. With 600 records, 100 injected positions and seeds
2/7/17, mean ROC-AUC (forest / baseline) was: digit preference .837/.996,
duplicates .732/.989, rounding .936/.790, uniform digits .968/.915, scaled
subset .564/.503 and amount outliers .550/.490. Mean precision@100 was
.293/.960, .150/.890, .607/.687, .767/.810, .237/.263 and .220/.263,
respectively, in the same order. Forest precision was **worse in every one**
of these scenarios; do not assume ML adds value. The clean-data index averaged
17.57; injected averages ranged from 17.57 (amount outliers) to 66.94 (digit
preference). These are synthetic scenarios, not real-world performance claims.

`Finding` records contain the numerical basis, severity, caveat references,
and deterministic written statement used by both UI and reports. They provide
the foundation for a later story layer; none is present here.

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
