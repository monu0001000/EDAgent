"""
test_streamlit_app.py
Smoke tests for streamlit_app.py using Streamlit's AppTest framework, which
runs the script headlessly (no browser needed) and lets us simulate widget
interactions like file uploads. These catch integration bugs that unit
tests on the individual modules wouldn't — e.g. a caching decorator
misbehaving, or the profile/chart data not making it through Streamlit's
render pipeline correctly.

Run with: python3 -m pytest test_streamlit_app.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import io
import numpy as np
import pandas as pd
import pytest

from streamlit.testing.v1 import AppTest


def _sample_csv_bytes() -> bytes:
    rng = np.random.default_rng(0)
    n = 100
    spend = rng.normal(50, 15, n)
    spend[0] = 5000
    df = pd.DataFrame({
        "customer_id": range(1, n + 1),
        "email": [f"user{i}@test.com" for i in range(n)],
        "age": rng.integers(18, 90, n).astype(float),
        "signup_date": pd.date_range("2023-01-01", periods=n, freq="D").astype(str),
        "plan": rng.choice(["basic", "pro", "enterprise"], n),
        "spend": spend,
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def test_app_loads_without_error_before_upload():
    at = AppTest.from_file("streamlit_app.py")
    at.run(timeout=15)
    assert not at.exception

def test_app_shows_upload_prompt_before_any_file():
    at = AppTest.from_file("streamlit_app.py")
    at.run(timeout=15)
    all_text = " ".join(m.value for m in at.markdown) + " ".join(i.value for i in at.info)
    assert "csv" in all_text.lower() or "upload" in all_text.lower()

def test_app_handles_csv_upload_end_to_end():
    """Drives an actual file through the uploader and verifies the
    profiler -> chart pipeline runs without raising inside Streamlit's
    execution model (this is the thing unit tests on profiler.py alone
    can't catch)."""
    at = AppTest.from_file("streamlit_app.py")
    at.run(timeout=15)

    uploader = at.file_uploader[0]
    uploader.set_value([("sample.csv", _sample_csv_bytes(), "text/csv")])
    at.run(timeout=30)

    assert not at.exception
    # Should have rendered stat cards, column details, and charts —
    # a bare "upload a file" page has only ~2 markdown elements.
    assert len(at.markdown) > 5

def test_report_button_disabled_without_api_key(monkeypatch):
    """The Generate report button must be disabled when no GEMINI_API_KEY
    is set, so a user can't click it and get a confusing API error."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    at = AppTest.from_file("streamlit_app.py")
    at.run(timeout=15)
    uploader = at.file_uploader[0]
    uploader.set_value([("sample.csv", _sample_csv_bytes(), "text/csv")])
    at.run(timeout=30)

    assert not at.exception
    assert len(at.button) == 1
    assert at.button[0].disabled is True

def test_report_mode_radio_has_both_options():
    at = AppTest.from_file("streamlit_app.py")
    at.run(timeout=15)
    uploader = at.file_uploader[0]
    uploader.set_value([("sample.csv", _sample_csv_bytes(), "text/csv")])
    at.run(timeout=30)

    assert not at.exception
    assert len(at.radio) == 1
    options = at.radio[0].options
    assert any("Agentic" in o for o in options)
    assert any("Single-shot" in o for o in options)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
