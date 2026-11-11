# Changes and upstream provenance

Upstream `benford_py` baseline: `0126c606ae9c27cba43e6dc83b73bb329f839ae4`.
Original BSD-3-Clause license, author attribution and original package retained.

Structural: root `pyproject.toml` builds both `fraudscope` and the existing
`benford` package in one editable install; the upstream Git repository remains
nested and is committed separately.

Upstream edits (additive, no legacy runtime behavior changed):
- `benford_py/benford/digits.py`: add four-mode extraction and normalized
  expected probabilities. The legacy last-two reference is 1/99 per bin;
  correcting it in place would change existing users' results, so the new
  toolkit uses the new API's 1/100 probabilities instead. Significant digit
  extraction now also supports subunit amounts without changing `get_digs`.
- `benford_py/tests/conftest.py`, `benford_py/tests/test_stats.py`: replace
  removed NumPy 2 `float_` alias with `float64` to restore test collection;
  assertions and expected behavior remain identical.
