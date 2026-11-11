# TASK: Build v1 of "fraudscope", a statistical anomaly-detection toolkit (Python + Jupyter)

## Context
This project directory already contains a local clone of `benford_py` (my fork).
Build the new toolkit in this same project directory, on top of that code. The
toolkit takes a tabular dataset and runs four statistical analyses: Benford's
law, Kolmogorov-Smirnov, Kuiper, and regression diagnostics. It produces plots
and a structured results object. This is v1: NO LLM, NO machine learning. Do
not add model-training code or LLM/API dependencies.

The tool detects ANOMALIES, not fraud. It must never output the words "fraud",
"rigged", or "fraudulent" in any result text, plot title, or report. Use
"deviation", "anomaly", "inconsistent with expected distribution". Every result
must carry a `caveats` list of innocent explanations for the deviation.

## PHASE 0: Recon (mandatory, do this before writing any code)
Read the `benford_py` directory thoroughly. Do not skim. Specifically:
1. Map the directory structure, packaging files (setup.py/pyproject/requirements),
   supported Python versions, dependencies, and the license.
2. Read the full public API: every function/class in the Benford module(s),
   their signatures, return types, and how digit extraction, expected
   distributions, and statistical tests are currently done.
3. Read its existing tests, notebooks, and docs. Run the existing test suite
   and record the baseline (what passes, what fails, what warns) BEFORE any
   change.
4. Note its code style, naming conventions, and any bugs, deprecated pandas/numpy
   usage, or limitations you notice that affect this project.
5. Record the current git commit hash of the clone (this is the upstream
   baseline, so my modifications can be separated from upstream later).
6. Reply with a recon summary: structure, API inventory, baseline test result,
   problems found, and a short plan stating (a) what you will reuse as-is,
   (b) what you will modify inside benford_py, and (c) what goes in new code.
   Then proceed without waiting for approval, unless you find something that
   contradicts this spec or a change that would break existing benford_py
   behavior for current users, in which case stop and ask.

## Rules for working with the codebase
- You MAY modify benford_py directly: extend it, refactor it, fix bugs, update
  deprecated APIs, add digit modes, add return fields. Prefer extending over
  rewriting; do not rewrite what already works.
- Keep a `CHANGES.md` at the project root logging every modification to
  benford_py: file, what changed, and why. Keep these edits in their own
  commits, separate from the new-package commits, with messages prefixed
  `benford_py:`.
- The existing benford_py test suite must keep passing after your changes
  (update a test only if the behavior change is intentional and logged in
  CHANGES.md).
- Preserve the original license and attribution. Do not remove authorship
  headers.
- Do not rewrite git history, force-push, or push to any remote. Local commits
  only.
- Do not delete files from benford_py; deprecate instead.
- You may modify any other part of the project codebase as needed, but keep
  every change within scope of this spec and log structural changes in
  CHANGES.md.

## Stack
- Python 3.10+ (or the benford_py minimum if higher), numpy, pandas, scipy,
  statsmodels, matplotlib (seaborn optional)
- Jupyter notebooks for demos; pytest for tests
- Plots: matplotlib, clean, labeled axes, saved as PNG when a path is given
- No other heavy dependencies. astropy allowed as a DEV-ONLY dependency, solely
  to cross-check the Kuiper implementation in tests.
- Reconcile packaging with the existing benford_py setup (use its build system
  if it has one; otherwise add `pyproject.toml`). Make `fraudscope` and
  `benford_py` both installable in one editable install.

## Layout (adapt to what Phase 0 reveals)
<project root>/
  benford_py/          # existing clone, modified in place where needed
  fraudscope/
    __init__.py
    results.py         # TestResult dataclass
    ingest.py          # load_csv/xlsx -> DataFrame, column selection, cleaning
    applicability.py   # checks whether a column suits each test
    benford.py         # thin layer over benford_py: adds TestResult output,
                       # applicability gate, record-level flags
    ks.py
    kuiper.py
    regression.py
    plots.py           # all plotting functions
    runner.py          # run_all(df, config) -> list[TestResult]
  tests/               # pytest for fraudscope (benford_py keeps its own tests)
  notebooks/
    01_benford_demo.ipynb
    02_ks_kuiper_demo.ipynb
    03_regression_demo.ipynb
    04_end_to_end.ipynb
  CHANGES.md
  README.md

## Core contract: TestResult (results.py)
A dataclass used by EVERY test:
- test_name: str
- column: str | None
- statistic: float | dict[str, float]
- p_value: float | None
- p_value_method: str   # "asymptotic" | "monte_carlo" | "none"
- effect_size: float | None   # e.g. MAD for Benford, D/V for KS/Kuiper
- n: int
- details: dict   # per-digit / per-bin deviations, expected vs observed, etc.
- flagged_indices: list[int]   # row indices contributing most to deviation
- caveats: list[str]
- warnings: list[str]   # e.g. "n too small", "applicability check failed"
Provide `.to_dict()` and `.to_json()`. All tests return this and nothing else,
so a reporting layer can consume them later.

## Module 1: applicability.py
Functions returning (ok: bool, reasons: list[str]):
- `benford_applicable(series)`: positive values; n >= 300 (warn below 1000);
  spans >= 3 orders of magnitude (max/min >= 1000, using non-zero positives);
  flag if a large share of values are identical or on a few round values; flag
  if data looks like assigned identifiers (fixed digit length, near-sequential).
- `continuous_applicable(series)`: numeric, n >= 30 for KS/Kuiper.
Failed checks must NOT block execution by default: run the test but attach the
reasons to `warnings`. Add a `strict=True` option that raises instead.

## Module 2: Benford (benford_py + fraudscope/benford.py)
Reuse benford_py for digit extraction and expected distributions. If it lacks
any of the following, add it INSIDE benford_py (logged in CHANGES.md), then
expose it through fraudscope/benford.py:
- Digit modes: first digit, second digit, first-two digits, last-two digits
  (expected uniform for last-two).
For each mode, return in `details`:
- observed vs expected proportions per digit
- per-digit z-statistics (with continuity correction)
- chi-square statistic + p-value
- MAD and Nigrini's conformity label (published first-digit thresholds:
  <0.006 close, <0.012 acceptable, <0.015 marginal, else nonconformity; look up
  and document thresholds for the other modes, citing the source in a docstring)
- `effect_size` = MAD, with a caveat that at large n chi-square flags nearly
  everything, so MAD is the primary measure.
Record-level flags: return indices of rows whose digit falls in the most
over-represented bins (top-k bins by z-score, configurable, default 3).
Always attach caveats: Benford may not hold for bounded or size-limited data;
deviations can come from rounding, price thresholds, or data-entry conventions.

## Module 3: ks.py
- `ks_one_sample(series, reference)`: reference is a scipy frozen distribution,
  a callable CDF, or the string "benford" (CDF on the log10-mantissa scale;
  document the transform clearly).
- `ks_two_sample(a, b)`.
- Compute D, D+, D- yourself (vectorized, from the sorted ECDF) AND the
  p-value. Use scipy's p-value for continuous references. For DISCRETE
  references (digit distributions) use a Monte Carlo p-value: simulate `n_sim`
  (default 10_000, seedable via `random_state`) samples of size n from the
  reference, compute the statistic for each, p = (1 + #{sim >= obs})/(1 + n_sim).
  Set `p_value_method` accordingly.
- Verify own D against `scipy.stats.ks_1samp` / `ks_2samp` in tests.

## Module 4: kuiper.py
Implement from scratch:
- V = D+ + D- from the sorted sample and a CDF.
- Asymptotic p-value: lam = (sqrt(n) + 0.155 + 0.24/sqrt(n)) * V;
  Q = 2 * sum_{j=1..J} (4 j^2 lam^2 - 1) * exp(-2 j^2 lam^2), J=100, clipped
  to [0, 1]; handle small lam (Q -> 1) without numerical blowups.
- Monte Carlo p-value option, same design as KS (needed for discrete refs).
- Two-sample Kuiper (effective n = n1*n2/(n1+n2)).
- Same reference options as KS ("benford", frozen scipy dist, callable CDF).
- Docstring: when Kuiper beats KS (circular/cyclic data such as time-of-day,
  tail and shift sensitivity).
- Tests: compare against `astropy.stats.kuiper` / `kuiper_two` on random
  samples (state the tolerance) and check p-value uniformity under the null
  via simulation.

## Module 5: regression.py
Inputs: DataFrame, `y` column, list of `x` columns, optional `group` column.
- OLS via statsmodels: coefficients, R^2, adj R^2, F-test, residual standard
  error.
- Robust fit: statsmodels RLM with Huber's T. Report both OLS and robust
  coefficients and warn when they differ materially.
- Per-row diagnostics: studentized residuals, leverage, Cook's distance,
  DFFITS. Flag rows exceeding standard cutoffs (|studentized resid| > 3,
  Cook's D > 4/n, leverage > 2(p+1)/n); return indices ordered by severity.
- Assumption checks: Breusch-Pagan, Shapiro-Wilk or Jarque-Bera on residuals
  (Shapiro limits at n>5000), Durbin-Watson if an order column is given.
  Put results in `details`; warn when violated.
- Group analysis: if `group` is given, per-group mean residual plus a one-way
  test (ANOVA or Kruskal-Wallis) on residuals across groups; flag groups with
  extreme mean residuals.
- Digit-level regression helper: regress observed on expected digit
  frequencies (from the Benford module): slope, intercept, R^2.
- Caveats: association is not causation; outliers can be legitimate; model
  misspecification produces false flags.

## Module 6: plots.py
Each function takes a TestResult (or raw data plus a result), returns a
matplotlib Figure, with optional `save_path`.
- `plot_benford(result)`: observed vs expected per digit with confidence bounds
  (expected +/- 1.96 sd) and flagged digits highlighted.
- `plot_ecdf_vs_reference(...)`: ECDF vs reference CDF with the max-D position
  (KS) or the D+ and D- points (Kuiper) marked.
- `plot_residuals(regression_result)`: residuals vs fitted, QQ plot, Cook's
  distance stem plot, leverage vs studentized residual with flagged points
  labeled.
- `plot_group_residuals(...)`: boxplot or dot plot by group.
- `plot_summary_dashboard(results)`: grid of key plots for a run.
Plot titles and labels must follow the wording rule.

## Module 7: runner.py + ingest.py
- `ingest.load(path, columns=None)`: CSV/XLSX via pandas, strips whitespace,
  coerces numerics, reports dropped/NaN counts.
- `run_all(df, config)`: config is a dataclass (columns to analyze, benford
  modes, reference distributions, regression spec, random_state, strict). Run
  applicable tests, collect `list[TestResult]`, never crash the whole run on one
  failing test (catch and record in warnings).
- Benjamini-Hochberg helper across the p-values in a run, stored as
  `details["p_adjusted"]` (statsmodels multipletests).

## Tests (pytest)
- Synthetic Benford-conforming data (e.g. 10**uniform or high-sigma lognormal):
  tests should NOT flag (check MAD label and p-values over several seeds).
- Injected anomalies: digit preference (extra 5s/9s), rounding to nearest 100,
  fabricated uniform digits, duplicated values, scaled subset: tests SHOULD
  detect them. Write a seeded helper that builds these datasets.
- Unit tests for each statistic vs scipy/statsmodels/astropy references.
- Edge cases: tiny n, zeros/negatives, NaNs, constant column, all-identical
  digits.
- Deterministic and offline. Run BOTH the benford_py suite and the fraudscope
  suite before each commit.

## Notebooks
Each runs top to bottom on synthetic data it generates (no external files).
04_end_to_end: load -> run_all -> summary dashboard -> inspect flagged rows.
Markdown cells explain each statistic in plain language and repeat the
"anomaly, not proof" caveat.

## Code standards
Type hints, docstrings with formulas and references, vectorized numpy (no row
loops), no global state, `random_state` on every stochastic function, ruff/black
formatting, small pure functions. Match benford_py's conventions inside
benford_py; use the standards above in new code. Do not add features beyond
this spec.

## Deliverable order (commit after each step, run both test suites first)
0. Recon summary (Phase 0) and baseline test results
1. Scaffold + TestResult + applicability + ingest + any benford_py fixes needed
   to proceed
2. fraudscope/benford.py (+ benford_py extensions) + tests + plot_benford
3. ks.py + kuiper.py + tests + ECDF plots
4. regression.py + tests + residual plots
5. runner.py + BH correction + dashboard plot
6. Notebooks + README (install, quickstart, wording/caveat policy, what each
   test is for and its limits) + final CHANGES.md

After each step, summarize what was built, which tests pass, and any decision
that deviates from this spec.