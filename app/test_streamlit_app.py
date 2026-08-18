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
    at = AppTest.from_file("streamlit_app.py")
    at.run(timeout=15)

    uploader = at.file_uploader[0]
    uploader.set_value([("sample.csv", _sample_csv_bytes(), "text/csv")])
    at.run(timeout=30)

    assert not at.exception
    assert len(at.markdown) > 5

def test_report_button_disabled_without_api_key(monkeypatch):
    """The Generate report button must be disabled when GROQ_API_KEY is
    not set, so a user can't click it and get a confusing API error.

    Uses setenv("") rather than delenv(): streamlit_app.py calls
    load_dotenv() on every AppTest run, and python-dotenv only fills in
    variables that are completely ABSENT from os.environ — so delenv()
    alone gets silently undone the moment a real .env file with a real
    key exists on the machine running the test (every local dev machine,
    just not CI). Setting it to an empty string keeps the key "present"
    from load_dotenv's point of view (so it won't touch it) while still
    being falsy for the app's own `if not os.environ.get(...)` check."""
    monkeypatch.setenv("GROQ_API_KEY", "")

    at = AppTest.from_file("streamlit_app.py")
    at.run(timeout=15)
    uploader = at.file_uploader[0]
    uploader.set_value([("sample.csv", _sample_csv_bytes(), "text/csv")])
    at.run(timeout=30)

    assert not at.exception
    buttons = {b.label: b for b in at.button}
    assert set(buttons) == {"Generate report", "Ask"}
    assert buttons["Generate report"].disabled is True
    assert buttons["Ask"].disabled is True

def test_report_button_enabled_with_api_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-test")

    at = AppTest.from_file("streamlit_app.py")
    at.run(timeout=15)
    uploader = at.file_uploader[0]
    uploader.set_value([("sample.csv", _sample_csv_bytes(), "text/csv")])
    at.run(timeout=30)

    assert not at.exception
    buttons = {b.label: b for b in at.button}
    assert buttons["Generate report"].disabled is False
    # "Ask" also requires a non-empty question, which isn't set here, so
    # it stays disabled even with a valid key — covered separately below.
    assert buttons["Ask"].disabled is True

def test_ask_button_enabled_once_api_key_and_question_are_both_present(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-test")

    at = AppTest.from_file("streamlit_app.py")
    at.run(timeout=15)
    uploader = at.file_uploader[0]
    uploader.set_value([("sample.csv", _sample_csv_bytes(), "text/csv")])
    at.run(timeout=30)

    at.text_input[0].set_value("is this a good deal?")
    at.run(timeout=15)

    assert not at.exception
    buttons = {b.label: b for b in at.button}
    assert buttons["Ask"].disabled is False

def test_ask_a_question_end_to_end_with_mocked_agent(monkeypatch):
    """Full click-through of the Q&A flow: type a question, click Ask,
    verify the mocked answer renders and gets kept in session state (so a
    second question doesn't wipe out the first one's answer)."""
    from unittest.mock import MagicMock, patch

    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-test")

    def _final_message(text):
        msg = MagicMock()
        msg.tool_calls = None
        msg.content = text
        response = MagicMock()
        response.choices = [MagicMock(message=msg)]
        return response

    at = AppTest.from_file("streamlit_app.py")
    at.run(timeout=15)
    uploader = at.file_uploader[0]
    uploader.set_value([("sample.csv", _sample_csv_bytes(), "text/csv")])
    at.run(timeout=30)

    at.text_input[0].set_value("is this a good deal?")
    at.run(timeout=15)

    with patch("groq_agent.Groq") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _final_message(
            "Based on 12 comparable rows, yes — likely a good deal."
        )
        ask_button = next(b for b in at.button if b.label == "Ask")
        ask_button.click().run(timeout=30)

    assert not at.exception
    assert any("Based on 12 comparable rows" in m.value for m in at.markdown)
    assert any("is this a good deal?" in m.value for m in at.markdown)

def test_report_mode_radio_has_both_options():
    at = AppTest.from_file("streamlit_app.py")
    at.run(timeout=15)
    uploader = at.file_uploader[0]
    uploader.set_value([("sample.csv", _sample_csv_bytes(), "text/csv")])
    at.run(timeout=30)

    assert not at.exception
    assert len(at.radio) == 1  # only the Report mode radio now — no provider picker
    mode_radio = at.radio[0]
    assert any("Agentic" in o for o in mode_radio.options)
    assert any("Single-shot" in o for o in mode_radio.options)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))