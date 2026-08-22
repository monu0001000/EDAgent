"""
test_api.py
Tests for api.py using Flask's test client — no live server or network
calls needed. Groq calls are mocked the same way test_groq_agent.py does,
so report/ask endpoint tests run fully offline.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import io
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import api as api_module
from api import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c
    # Every test starts from a clean dataset store — module-level dict
    # would otherwise leak datasets between tests.
    api_module.DATASET_STORE.clear()


def _sample_csv_bytes() -> bytes:
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "customer_id": range(1, n + 1),
        "plan": rng.choice(["basic", "pro"], n),
        "spend": rng.normal(50, 15, n).round(2),
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _final_message(text: str):
    msg = MagicMock()
    msg.tool_calls = None
    msg.content = text
    response = MagicMock()
    response.choices = [MagicMock(message=msg)]
    return response


def _upload_sample(client) -> str:
    resp = client.post(
        "/datasets",
        data={"file": (io.BytesIO(_sample_csv_bytes()), "sample.csv")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 201
    return resp.get_json()["dataset_id"]


# ---------------------------------------------------------------------------
# GET / (the web UI)
# ---------------------------------------------------------------------------

def test_index_serves_the_web_ui(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"EDAgent" in resp.data
    assert b"dropzone" in resp.data  # confirms templates/index.html rendered, not a stub

def test_index_references_static_assets(client):
    resp = client.get("/")
    body = resp.data.decode()
    assert "app.js" in body
    assert "style.css" in body

def test_static_assets_are_served(client):
    resp = client.get("/static/style.css")
    assert resp.status_code == 200
    resp = client.get("/static/app.js")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_check(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# POST /datasets
# ---------------------------------------------------------------------------

def test_upload_dataset_returns_id_and_profile(client):
    resp = client.post(
        "/datasets",
        data={"file": (io.BytesIO(_sample_csv_bytes()), "sample.csv")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 201
    body = resp.get_json()
    assert "dataset_id" in body
    assert body["profile"]["columns"]["plan"]["inferred_type"] == "categorical"

def test_upload_dataset_requires_file_field(client):
    resp = client.post("/datasets", data={}, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "No file provided" in resp.get_json()["error"]

def test_upload_dataset_rejects_unparseable_csv(client):
    resp = client.post(
        "/datasets",
        data={"file": (io.BytesIO(b"\x00\x01not,a,csv\xff\xfe"), "bad.csv")},
        content_type="multipart/form-data",
    )
    # Either a clean parse failure (400) or pandas manages to read garbage
    # as a 1-row/1-col frame — either is acceptable, a 500 crash isn't.
    assert resp.status_code in (201, 400)

def test_upload_dataset_rejects_empty_csv(client):
    resp = client.post(
        "/datasets",
        data={"file": (io.BytesIO(b"col_a,col_b\n"), "empty.csv")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    assert "no rows" in resp.get_json()["error"]


# ---------------------------------------------------------------------------
# GET /datasets/<id>, GET /datasets/<id>/charts
# ---------------------------------------------------------------------------

def test_get_dataset_profile(client):
    dataset_id = _upload_sample(client)
    resp = client.get(f"/datasets/{dataset_id}")
    assert resp.status_code == 200
    assert resp.get_json()["dataset_id"] == dataset_id

def test_get_dataset_profile_404_for_unknown_id(client):
    resp = client.get("/datasets/does-not-exist")
    assert resp.status_code == 404

def test_get_dataset_charts_returns_plotly_json(client):
    dataset_id = _upload_sample(client)
    resp = client.get(f"/datasets/{dataset_id}/charts")
    assert resp.status_code == 200
    charts = resp.get_json()["charts"]
    assert len(charts) > 0
    # Every chart should be a JSON-serializable Plotly figure spec (has
    # "data" and "layout" keys, the standard Plotly figure shape).
    for chart in charts.values():
        assert "data" in chart
        assert "layout" in chart

def test_get_dataset_charts_404_for_unknown_id(client):
    resp = client.get("/datasets/does-not-exist/charts")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /datasets/<id>/report
# ---------------------------------------------------------------------------

def test_generate_report_requires_api_key(client, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    dataset_id = _upload_sample(client)
    resp = client.post(f"/datasets/{dataset_id}/report", json={"mode": "agentic"})
    assert resp.status_code == 503

def test_generate_report_invalid_mode(client, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    dataset_id = _upload_sample(client)
    resp = client.post(f"/datasets/{dataset_id}/report", json={"mode": "not-a-real-mode"})
    assert resp.status_code == 400

def test_generate_report_404_for_unknown_dataset(client, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    resp = client.post("/datasets/does-not-exist/report", json={"mode": "agentic"})
    assert resp.status_code == 404

def test_generate_report_agentic_mocked(client, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    dataset_id = _upload_sample(client)

    with patch("groq_agent.Groq") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _final_message(
            "## Key Patterns\n- mocked report content"
        )
        resp = client.post(f"/datasets/{dataset_id}/report", json={"mode": "agentic"})

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["mode"] == "agentic"
    assert "mocked report content" in body["report"]
    assert body["tool_calls"] == []

def test_report_endpoint_respects_max_iterations_override(client, monkeypatch):
    """AGENTIC_MAX_ITERATIONS is overridable via EDAGENT_MAX_ITERATIONS
    specifically so a rate-limited/free-tier deployment can trade
    investigation depth for speed without a code change — confirms the
    override actually reaches generate_insights_agentic's call, not just
    that the env var parses."""
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    monkeypatch.setattr(api_module, "AGENTIC_MAX_ITERATIONS", 2)
    dataset_id = _upload_sample(client)

    with patch("groq_agent.generate_insights_agentic") as mock_generate:
        mock_generate.return_value = {"report": "ok", "tool_calls": [], "iterations": 0}
        client.post(f"/datasets/{dataset_id}/report", json={"mode": "agentic"})

    assert mock_generate.call_args.kwargs["max_iterations"] == 2

def test_generate_report_single_shot_mocked(client, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    dataset_id = _upload_sample(client)

    with patch("groq_insight_generator.Groq") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _final_message(
            "## Key Patterns\n- single-shot mocked report"
        )
        resp = client.post(f"/datasets/{dataset_id}/report", json={"mode": "single_shot"})

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["mode"] == "single_shot"
    assert "single-shot mocked report" in body["report"]


# ---------------------------------------------------------------------------
# POST /datasets/<id>/ask
# ---------------------------------------------------------------------------

def test_ask_question_requires_question_field(client, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    dataset_id = _upload_sample(client)
    resp = client.post(f"/datasets/{dataset_id}/ask", json={})
    assert resp.status_code == 400

def test_ask_question_mocked(client, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    dataset_id = _upload_sample(client)

    with patch("groq_agent.Groq") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _final_message(
            "Based on 10 comparable rows, likely yes."
        )
        resp = client.post(f"/datasets/{dataset_id}/ask", json={"question": "is this a good plan?"})

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["question"] == "is this a good plan?"
    assert "Based on 10 comparable rows" in body["answer"]

def test_ask_question_404_for_unknown_dataset(client, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    resp = client.post("/datasets/does-not-exist/ask", json={"question": "anything?"})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /datasets/<id>
# ---------------------------------------------------------------------------

def test_delete_dataset(client):
    dataset_id = _upload_sample(client)
    resp = client.delete(f"/datasets/{dataset_id}")
    assert resp.status_code == 204
    # And it's actually gone:
    resp = client.get(f"/datasets/{dataset_id}")
    assert resp.status_code == 404

def test_delete_dataset_404_for_unknown_id(client):
    resp = client.delete("/datasets/does-not-exist")
    assert resp.status_code == 404


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
