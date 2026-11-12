"""Optional Streamlit smoke test (requires the app extra)."""

from pathlib import Path

import pytest


def test_sample_smoke():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "app/main.py"), default_timeout=120
    ).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert app.tabs
    assert set(app.multiselect[0].value) == {"amount", "sequence"}
    assert {result.column for result in app.session_state["bundle"].results} == {
        "amount",
        "sequence",
    }
