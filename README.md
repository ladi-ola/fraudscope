# fraudscope

Explore statistical anomalies in tabular data using a local web app or a Python API.

> **An anomaly is not proof of wrongdoing.** Scores rank records within the
> uploaded dataset; they are not probabilities or predictions of intent.

## Get started

**Web app** — Python 3.10+:

```bash
python -m pip install -e '.[app]'
streamlit run app/main.py
```

1. Upload a **CSV/XLSX** file, or click **Try with sample data**.
2. All numeric columns are selected by default. Leave them selected, or choose
   the ones you want to inspect. Non-numeric columns remain in the data profile.
3. Optionally choose digit modes, a reference comparison, a regression model,
   and the Benford chart band. Click **Run analysis**.
4. Review results by tab. Download **HTML**, **PDF**, or `findings.json` from
   the **Report** tab.

The Benford band can be any whole-number level from **1% to 99%** (default:
95%). A narrower band colors more bars on the chart. Changing it **does not**
change test results, written findings, or flagged records; each band applies
to one digit at a time, not all digits together.

### Screenshots

_Placeholder: Overview and Report tab screenshots._

## How it fits together

```text
CSV / XLSX / built-in sample
            │  processed in memory
            ▼
        Streamlit app ──────────────┐
            │                      │
            ▼                      │ displays
    run_analysis(df, config)        │
            │                      │
            ├── Benford, KS, Kuiper, regression
            ├── optional anomaly scoring
            └── rule-based Findings
                    │
                    ▼
             AnalysisBundle ────────┘
                    │
                    └── HTML report / PDF report / findings.json
```

The app calls the same analysis service used to build the reports. **Findings**
hold the numbers, severity, caveats, and deterministic written summaries; the
app and reports read the same results. This structure can support a later story
layer, but none is included now.

## Reading the results

| Check | What it looks for | Keep in mind |
| --- | --- | --- |
| **Benford** | Differences between observed and expected digit shares. Supports first, second, first-two, and last-two digits. | Bounded values, identifiers, prices, and rounding can explain differences. Last-two digits have no published conformity thresholds. |
| **KS** | The largest gap between a sample's cumulative distribution and its chosen reference. | Choose a reference independently of the sample when possible. |
| **Kuiper** | Gaps both above and below the reference; useful for shifts and circular data. | Small or discrete samples may need simulation for reliable p-values. |
| **Regression** | Unusual prediction errors, influential rows, and differences between fitted relationships. | A marked row can be genuine; missing predictors or model assumptions may explain it. |

**Flags are starting-at-zero row positions** in the uploaded data, not DataFrame
index labels. They identify rows to review, not wrongdoing. Each result includes
warnings and possible innocent explanations. Where available, p-values are
adjusted across the run for multiple checks.

### Anomaly scoring: what it does *not* do

Isolation Forest ranks records **relative to other records in the same
dataset**. The 0–100 dataset index summarizes review share and distribution
differences. Neither value is a probability. A score above the default 0.90
threshold selects roughly the top 10% of records even for clean data.

There is **no fixed minimum row count**. Scoring is unavailable if there are no
records or no variable numeric features. Missing features are imputed using
medians and given missing-value indicators. Very small samples can produce
unstable rankings; the report gives the actual counts, value ranges, and a
sample-size-based uncertainty guide.

### What synthetic validation found

`notebooks/05_ml_validation.ipynb` compares Isolation Forest with a **weighted
tests-only flag baseline**. Each scenario below used 600 records, 100 altered rows,
and seeds 2, 7, and 17. Values are averages:

| Alteration | ROC-AUC: forest / baseline | Precision@100: forest / baseline |
| --- | ---: | ---: |
| Digit preference | .837 / .996 | .293 / .960 |
| Duplicates | .732 / .989 | .150 / .890 |
| Rounding | .936 / .790 | .607 / .687 |
| Uniform digits | .968 / .915 | .767 / .810 |
| Scaled subset | .564 / .503 | .237 / .263 |
| Amount outliers | .550 / .490 | .220 / .263 |
| Fitted-regression outliers (separate scenario) | .900 / .681 | .550 / .490 |

**The forest's precision was worse than the baseline in all six amount-based
scenarios.** It did better in the separate fitted-regression scenario; do not
assume it adds value for other data. For the six amount-based scenarios, the
clean-data index averaged 17.57 and altered-data averages ranged from 17.57
(amount outliers) to 66.94 (digit preference). In the fitted-regression
scenario, both clean and altered datasets scored 29.46: the index does not
include regression flag rate, a known blind spot. These are **synthetic tests,
not real-world performance estimates**.

<details>
<summary>Index formula and statistical notes</summary>

The index is `100 × (0.2 × review share + 0.5 × mean normalized effect +
0.3 × mean adjusted significance)`. Effect sizes are clipped at Benford MAD
`.015/.012/.0022/.01` (first/second/first-two/last-two), KS D `.2`, and Kuiper
V `.3`. Significance for eligible distribution tests is
`max(0, 1 - adjusted_p / .05)`. These are descriptive design scales, **not
universal statistical cutoffs**. Regression F-test significance is excluded
because it measures association, not anomalous records. Nigrini's MAD
guidelines inform the first three digit modes; last-two and regression flag-rate
severity bands are descriptive only.

`reference="benford"` for KS/Kuiper compares the fractional `log10(x)` of
**positive values** with a uniform reference. This is different from comparing
discrete digit frequencies. Discrete references can be supplied as
`(sorted_bins, probabilities)` and use seeded Monte Carlo p-values. Kuiper's
asymptotic p-value is approximate; use simulation/permutation for small or
discrete data. Scaling a genuinely Benford-distributed subset can preserve its
leading digits, so digit tests alone may miss that change.

</details>

## Python quickstart

The original statistical API works without Streamlit:

```python
import numpy as np
import pandas as pd

from fraudscope.runner import Config, run_all

rng = np.random.default_rng(2)
df = pd.DataFrame({
    "amount": 10 ** rng.uniform(1, 5, 2_000),
    "sequence": np.arange(2_000),
})

results = run_all(df, Config(
    columns=["amount"],
    benford_modes=["first", "second", "first_two", "last_two"],
    references={"amount": "benford"},
    random_state=2,
))
print([result.to_dict() for result in results])
```

For a full `AnalysisBundle` with profile, optional scoring, Findings, and
report-ready figures, use `fraudscope.service.run_analysis(df, config)`.
`fraudscope.ingest.load("data.csv")` also accepts XLSX, trims whitespace,
converts fully numeric text columns, retains rows, and records cleaning counts
in `df.attrs["cleaning"]`. Direct tests with `strict=True` raise on unsuitable
data; the runner isolates failures so one check does not end a whole run.

## Privacy, limits, and attribution

- Uploaded data stays **in memory**. The app does not send it to an external
  service or save it to disk. HTML and PDF reports are self-contained.
- There is no application-specific row or file-size cutoff. Available memory
  and Streamlit's configurable `server.maxUploadSize` setting still apply (the
  Streamlit default is 200 MB). To raise the upload limit, for example:
  `streamlit run app/main.py --server.maxUploadSize 500`.
- The bundled Source Code Pro font (Adobe, SIL Open Font License 1.1) is used
  offline in the app and reports, with sans-serif as fallback. See
  `fraudscope/report/fonts/LICENSE.md`.
- One editable install exposes both this package and the local `benford_py`
  package. Its BSD-3-Clause license remains in `benford_py/LICENSE.txt`.
  Upstream provenance and changes are documented in `CHANGES.md`.

## Tests and examples

```bash
python -m pip install -e '.[app,dev]'
python -m pytest -q tests benford_py/tests
```

The `notebooks/` directory contains four statistical demos and the ML
validation notebook.
