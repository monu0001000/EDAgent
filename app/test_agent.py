"""
test_agent.py
Tests the agentic loop logic (agent.py) using a mocked Gemini client, so we
can verify message formatting, tool execution wiring, and the
max_iterations safety cap without needing a live API key or making real
network calls.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from agent import generate_insights_agentic, _generate_with_retry
from google.genai import errors


def _final_response(text):
    """A Gemini response with no function calls — the model's final answer."""
    r = MagicMock()
    r.function_calls = []
    r.text = text
    return r


def _tool_call_response(expression, call_name="run_pandas_query"):
    """A Gemini response requesting one function call."""
    fc = MagicMock()
    fc.name = call_name
    fc.args = {"expression": expression}

    r = MagicMock()
    r.function_calls = [fc]
    r.text = None
    r.candidates = [MagicMock()]
    r.candidates[0].content = MagicMock(name="model_turn_content")
    return r


@pytest.fixture
def df():
    return pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "x", "y", "z"]})

@pytest.fixture
def profile():
    return {"shape": {"rows": 5, "cols": 2}, "columns": {}}


def test_agent_returns_immediately_if_no_tool_use(df, profile):
    """If the model's first response has no function_calls, the loop should
    return right away without trying to execute any tools."""
    mock_response = _final_response("## Key Patterns\n- nothing notable")

    with patch("agent.genai.Client") as MockClient:
        MockClient.return_value.models.generate_content.return_value = mock_response
        result = generate_insights_agentic(df, profile, verbose=False)

    assert "Key Patterns" in result["report"]
    assert result["tool_calls"] == []
    assert result["iterations"] == 0

def test_agent_executes_tool_call_and_continues(df, profile):
    """First response requests a tool call; second response gives the final
    answer. Verify the tool was actually run through the sandbox and its
    result was fed back before the final answer was produced."""
    first_response = _tool_call_response("df['a'].mean()")
    second_response = _final_response("## Key Patterns\n- mean of a is 3.0")

    with patch("agent.genai.Client") as MockClient:
        mock_generate = MockClient.return_value.models.generate_content
        mock_generate.side_effect = [first_response, second_response]
        result = generate_insights_agentic(df, profile, verbose=False)

    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["expression"] == "df['a'].mean()"
    assert result["tool_calls"][0]["result"] == "3.0"
    assert "mean of a is 3.0" in result["report"]

def test_agent_rejects_unsafe_query_but_continues_loop(df, profile):
    """If the model requests an unsafe query, the sandbox should reject it
    and the rejection message should be fed back as the function_response
    (not crash the whole loop) — this both protects the sandbox and lets the
    model see that its query was rejected and try something else."""
    first_response = _tool_call_response("open('/etc/passwd')")
    second_response = _final_response("## Key Patterns\n- could not inspect further")

    with patch("agent.genai.Client") as MockClient:
        mock_generate = MockClient.return_value.models.generate_content
        mock_generate.side_effect = [first_response, second_response]
        result = generate_insights_agentic(df, profile, verbose=False)

    assert "rejected" in result["tool_calls"][0]["result"].lower()

def test_agent_respects_max_iterations_cap(df, profile):
    """If the model keeps requesting tool calls forever, the loop must stop
    at max_iterations and force a final answer rather than looping forever
    (which would be both a cost risk and a reliability bug)."""
    looping_response = _tool_call_response("df['a'].sum()")
    forced_final_response = _final_response("## Key Patterns\n- forced final answer")

    with patch("agent.genai.Client") as MockClient:
        mock_generate = MockClient.return_value.models.generate_content
        # With max_iterations=3, the loop makes range(4) = 4 calls (all
        # returning a tool call here, simulating a model that never stops
        # asking for more tool calls), then 1 forced final call outside
        # the loop = 5 calls total.
        mock_generate.side_effect = [looping_response] * 4 + [forced_final_response]
        result = generate_insights_agentic(df, profile, max_iterations=3, verbose=False)

    assert result["iterations"] == 3
    assert len(result["tool_calls"]) == 4  # one per loop iteration
    assert "forced final answer" in result["report"]

def test_agent_appends_model_turn_and_tool_response_correctly(df, profile):
    """Verify the contents list sent back to the API on the second call has
    the correct shape: the model's own turn (with the function_call), then
    a follow-up turn containing the function_response. Note: role is "user"
    here, not "tool" — the live Gemini API rejects role="tool" with a 400
    (some SDK docs show "tool" but it isn't actually accepted)."""
    first_response = _tool_call_response("df['a'].mean()")
    second_response = _final_response("done")

    with patch("agent.genai.Client") as MockClient:
        mock_generate = MockClient.return_value.models.generate_content
        mock_generate.side_effect = [first_response, second_response]
        generate_insights_agentic(df, profile, verbose=False)

    second_call_kwargs = mock_generate.call_args_list[1].kwargs
    contents = second_call_kwargs["contents"]

    # index 0 = initial user turn, index 1 = model's function_call turn,
    # index 2 = our follow-up turn with the function_response
    assert contents[1] is first_response.candidates[0].content
    assert contents[2].role == "user"
    assert len(contents[2].parts) == 1


# ---------- Rate limit retry ----------

def test_retry_succeeds_after_one_429(monkeypatch):
    """A single 429 should be retried (after a short-circuited sleep) and
    succeed on the next attempt, rather than propagating as a failure."""
    monkeypatch.setattr("agent.time.sleep", lambda s: None)  # skip real waiting in tests

    mock_client = MagicMock()
    final_response = MagicMock()
    final_response.text = "ok"
    mock_client.models.generate_content.side_effect = [
        errors.ClientError(429, {"error": {"message": "rate limited"}}, None),
        final_response,
    ]

    result = _generate_with_retry(mock_client, "gemini-flash-latest", [], None, verbose=False)
    assert result.text == "ok"
    assert mock_client.models.generate_content.call_count == 2

def test_retry_gives_up_after_max_retries(monkeypatch):
    """If every attempt is rate limited, the error should eventually
    propagate rather than retrying forever."""
    monkeypatch.setattr("agent.time.sleep", lambda s: None)

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = errors.ClientError(
        429, {"error": {"message": "rate limited"}}, None
    )

    with pytest.raises(errors.ClientError):
        _generate_with_retry(mock_client, "gemini-flash-latest", [], None, max_retries=3, verbose=False)
    assert mock_client.models.generate_content.call_count == 3

def test_non_429_errors_are_not_retried(monkeypatch):
    """A non-rate-limit error (e.g. 400 bad request) should propagate
    immediately without wasting retries/time on it."""
    monkeypatch.setattr("agent.time.sleep", lambda s: None)

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = errors.ClientError(
        400, {"error": {"message": "bad request"}}, None
    )

    with pytest.raises(errors.ClientError):
        _generate_with_retry(mock_client, "gemini-flash-latest", [], None, verbose=False)
    assert mock_client.models.generate_content.call_count == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
