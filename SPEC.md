# v1 implementation checklist

- [x] Recon: map legacy API, packaging, tests, demo, license, upstream commit; attempt baseline.
- [x] Install dependencies and record executable baseline before changing `benford_py`.
- [x] Scaffold one editable distribution, result contract, applicability checks, ingestion.
- [x] Add compatible digit helpers upstream; four Benford modes, tests, figure.
- [x] One/two-sample KS and Kuiper, continuous and discrete calibration, tests, ECDF figures.
- [x] OLS/Huber diagnostics, assumptions, group comparison, plots and tests.
- [x] Fault-isolated runner with BH correction and dashboard.
- [x] Four offline notebooks, README, change log, full test pass and local commits.

Each checked item is verified with both test suites before committing. Row flags
are positions in the input frame (integer positional indices), not index labels.
