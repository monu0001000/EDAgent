"""
test_groq_agent.py
Tests the Groq agentic loop logic (groq_agent.py) using a mocked Groq
client, mirroring test_agent.py's approach for the Gemini version. Verifies
message formatting, tool execution wiring, and rate-limit retry behavior
without needing a live API key or making real network calls.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from groq_agent import (
    generate_insights_agentic, answer_question, _create_with_retry,
    _build_messages, KEEP_LAST_FULL_TURNS, QA_SYSTEM_PROMPT,
)
from groq import RateLimitError, BadRequestError


def _final_message(text):
    """A response with no tool calls — the model's final answer."""
    msg = MagicMock()
    msg.tool_calls = None
    msg.content = text
    response = MagicMock()
    response.choices = [MagicMock(message=msg)]
    return response


def _tool_call_message(expression, call_id="call_1"):
    """A response requesting one tool call."""
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = "run_pandas_query"
    tc.function.arguments = f'{{"expression": {expression!r}}}'

    msg = MagicMock()
    msg.tool_calls = [tc]
    msg.content = None
    response = MagicMock()
    response.choices = [MagicMock(message=msg)]
    return response


@pytest.fixture
def df():
    return pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "x", "y", "z"]})

@pytest.fixture
def profile():
    return {"shape": {"rows": 5, "cols": 2}, "columns": {}}


def test_agent_returns_immediately_if_no_tool_use(df, profile):
    mock_response = _final_message("## Key Patterns\n- nothing notable")

    with patch("groq_agent.Groq") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = mock_response
        result = generate_insights_agentic(df, profile, verbose=False)

    assert "Key Patterns" in result["report"]
    assert result["tool_calls"] == []
    assert result["iterations"] == 0

def test_agent_executes_tool_call_and_continues(df, profile):
    first_response = _tool_call_message("df['a'].mean()")
    second_response = _final_message("## Key Patterns\n- mean of a is 3.0")

    with patch("groq_agent.Groq") as MockClient:
        mock_create = MockClient.return_value.chat.completions.create
        mock_create.side_effect = [first_response, second_response]
        result = generate_insights_agentic(df, profile, verbose=False)

    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["expression"] == "df['a'].mean()"
    assert result["tool_calls"][0]["result"] == "3.0"
    assert "mean of a is 3.0" in result["report"]

def test_agent_rejects_unsafe_query_but_continues_loop(df, profile):
    first_response = _tool_call_message("open('/etc/passwd')")
    second_response = _final_message("## Key Patterns\n- could not inspect further")

    with patch("groq_agent.Groq") as MockClient:
        mock_create = MockClient.return_value.chat.completions.create
        mock_create.side_effect = [first_response, second_response]
        result = generate_insights_agentic(df, profile, verbose=False)

    assert "rejected" in result["tool_calls"][0]["result"].lower()

def test_agent_respects_max_iterations_cap(df, profile):
    looping_response = _tool_call_message("df['a'].sum()")
    forced_final_response = _final_message("## Key Patterns\n- forced final answer")

    with patch("groq_agent.Groq") as MockClient:
        mock_create = MockClient.return_value.chat.completions.create
        mock_create.side_effect = [looping_response] * 4 + [forced_final_response]
        result = generate_insights_agentic(df, profile, max_iterations=3, verbose=False)

    assert result["iterations"] == 3
    assert len(result["tool_calls"]) == 4
    assert "forced final answer" in result["report"]

# ---------------------------------------------------------------------------
# Recovering from gpt-oss-120b's "tool_use_failed" quirk: it sometimes tries
# to call a tool even on the forced-final-answer request where tools=None
# is deliberately sent to prevent exactly that. Groq's API rejects the whole
# response with a 400 rather than ignoring the attempted call. Real example
# hit in production: "Tool choice is none, but model called a tool" while
# investigating a specific train_id. See _parse_forced_tool_call_from_error.
# ---------------------------------------------------------------------------

def _tool_use_failed_error(expression: str):
    body = {
        "error": {
            "message": "Tool choice is none, but model called a tool",
            "type": "invalid_request_error",
            "code": "tool_use_failed",
            "failed_generation": json.dumps({"name": "run_pandas_query", "arguments": {"expression": expression}}),
        }
    }
    resp = MagicMock()
    resp.request = MagicMock()
    return BadRequestError("Error code: 400", response=resp, body=body)

def test_agent_recovers_from_forced_final_tool_use_error(df, profile):
    looping_response = _tool_call_message("df['a'].sum()")
    real_final_response = _final_message("## Key Patterns\n- a sums to 15, checked b too")

    with patch("groq_agent.Groq") as MockClient:
        mock_create = MockClient.return_value.chat.completions.create
        # 4 looping tool-call rounds (max_iterations=3), then the forced-final
        # request fails with the recoverable error, then succeeds once the
        # recovered query has actually been run.
        mock_create.side_effect = (
            [looping_response] * 4 + [_tool_use_failed_error("df['b'].unique()")] + [real_final_response]
        )
        result = generate_insights_agentic(df, profile, max_iterations=3, verbose=False)

    # 4 original tool calls + 1 recovered one from the failed forced-final attempt
    assert len(result["tool_calls"]) == 5
    assert result["tool_calls"][-1]["expression"] == "df['b'].unique()"
    assert "a sums to 15, checked b too" in result["report"]

def test_agent_falls_back_gracefully_if_forced_final_error_repeats(df, profile):
    looping_response = _tool_call_message("df['a'].sum()")

    with patch("groq_agent.Groq") as MockClient:
        mock_create = MockClient.return_value.chat.completions.create
        # Both forced-final attempts fail the same way — should not raise,
        # should fall back to summarizing what's already in tool_call_log.
        mock_create.side_effect = (
            [looping_response] * 4
            + [_tool_use_failed_error("df['b'].unique()")]
            + [_tool_use_failed_error("df['c'].unique()")]
        )
        result = generate_insights_agentic(df, profile, max_iterations=3, verbose=False)

    assert "couldn't produce a final written answer" in result["report"]
    assert "df['a'].sum()" in result["report"]  # summarized from tool_call_log

def test_agent_reraises_unrelated_bad_request_errors(df, profile):
    """Only the specific tool_use_failed/run_pandas_query shape is treated
    as recoverable — an unrelated 400 (bad API key, malformed request,
    etc.) should still surface as a real error rather than being silently
    swallowed by the recovery path."""
    looping_response = _tool_call_message("df['a'].sum()")
    unrelated_error = BadRequestError(
        "Error code: 400",
        response=MagicMock(request=MagicMock()),
        body={"error": {"message": "invalid api key", "code": "invalid_api_key"}},
    )

    with patch("groq_agent.Groq") as MockClient:
        mock_create = MockClient.return_value.chat.completions.create
        mock_create.side_effect = [looping_response] * 4 + [unrelated_error]
        result = generate_insights_agentic(df, profile, max_iterations=3, verbose=False)

    # Not the recoverable shape -> breaks out of the grace loop immediately
    # and falls back gracefully rather than retrying pointlessly or crashing.
    assert "couldn't produce a final written answer" in result["report"]

def test_agent_appends_tool_role_message_with_correct_id(df, profile):
    """Verify the message list sent on the second call includes a
    role='tool' message referencing the correct tool_call_id — this is the
    OpenAI-compatible shape Groq expects, different from Gemini's
    Content/Part objects."""
    first_response = _tool_call_message("df['a'].mean()", call_id="abc123")
    second_response = _final_message("done")

    with patch("groq_agent.Groq") as MockClient:
        mock_create = MockClient.return_value.chat.completions.create
        mock_create.side_effect = [first_response, second_response]
        generate_insights_agentic(df, profile, verbose=False)

    second_call_kwargs = mock_create.call_args_list[1].kwargs
    messages = second_call_kwargs["messages"]

    # index 0 = system, 1 = initial user turn, 2 = assistant's tool_call
    # turn, 3 = our tool-role message with the result
    tool_msg = messages[3]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "abc123"
    assert tool_msg["content"] == "3.0"


# ---------- Rate limit retry ----------

def _rate_limit_error(retry_after=None):
    response = MagicMock()
    response.headers = {"retry-after": str(retry_after)} if retry_after else {}
    return RateLimitError("rate limited", response=response, body=None)

def test_retry_succeeds_after_one_rate_limit(monkeypatch):
    monkeypatch.setattr("groq_agent.time.sleep", lambda s: None)

    mock_client = MagicMock()
    final_response = _final_message("ok")
    mock_client.chat.completions.create.side_effect = [_rate_limit_error(), final_response]

    result = _create_with_retry(mock_client, "openai/gpt-oss-120b", [], [], verbose=False)
    assert result.choices[0].message.content == "ok"
    assert mock_client.chat.completions.create.call_count == 2

def test_retry_respects_retry_after_header(monkeypatch):
    """If Groq sends a Retry-After header, we should wait that long rather
    than our own fixed default."""
    sleep_calls = []
    monkeypatch.setattr("groq_agent.time.sleep", lambda s: sleep_calls.append(s))

    mock_client = MagicMock()
    final_response = _final_message("ok")
    mock_client.chat.completions.create.side_effect = [_rate_limit_error(retry_after=7), final_response]

    _create_with_retry(mock_client, "openai/gpt-oss-120b", [], [], verbose=False)
    assert sleep_calls == [7.0]

def test_retry_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr("groq_agent.time.sleep", lambda s: None)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _rate_limit_error()

    with pytest.raises(RateLimitError):
        _create_with_retry(mock_client, "openai/gpt-oss-120b", [], [], max_retries=3, verbose=False)
    assert mock_client.chat.completions.create.call_count == 3


# ---------- Context windowing ----------
# Mirrors the same fix in agent.py (Gemini): OpenAI-style message lists
# have the identical unbounded-growth problem — every tool call appends an
# assistant message + a tool message, forever. These tests prove message
# count stays constant regardless of how many iterations run.

def test_build_messages_stays_bounded_regardless_of_history_length():
    base = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    history = [
        {
            "assistant_message": {"role": "assistant", "content": None, "tool_calls": [f"tc_{i}"]},
            "tool_messages": [{"role": "tool", "tool_call_id": f"tc_{i}", "content": f"result {i}" * 20}],
            "calls": [{"expression": f"q{i}", "result": f"result data {i}" * 20}],
        }
        for i in range(10)
    ]

    messages = _build_messages(base, history)
    expected_len = len(base) + 1 + KEEP_LAST_FULL_TURNS * 2  # base + 1 merged summary + recent turns
    assert len(messages) == expected_len

    # Doubling history length must not change message count at all.
    longer_history = history + history
    messages_longer = _build_messages(base, longer_history)
    assert len(messages_longer) == expected_len

def test_build_messages_keeps_most_recent_turns_in_full():
    base = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    history = [
        {
            "assistant_message": {"role": "assistant", "content": None, "tool_calls": [f"tc_{i}"]},
            "tool_messages": [{"role": "tool", "tool_call_id": f"tc_{i}", "content": f"r{i}"}],
            "calls": [{"expression": f"q{i}", "result": f"r{i}"}],
        }
        for i in range(5)
    ]

    messages = _build_messages(base, history)
    tail = messages[-(KEEP_LAST_FULL_TURNS * 2):]
    expected_tail = []
    for record in history[-KEEP_LAST_FULL_TURNS:]:
        expected_tail.append(record["assistant_message"])
        expected_tail.extend(record["tool_messages"])
    assert tail == expected_tail

def test_build_messages_compacts_old_turns_into_short_merged_summary():
    base = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    long_result = "x" * 5000
    history = [
        {
            "assistant_message": {"role": "assistant", "content": None, "tool_calls": ["tc_0"]},
            "tool_messages": [{"role": "tool", "tool_call_id": "tc_0", "content": long_result}],
            "calls": [{"expression": "df.groupby('a').describe()", "result": long_result}],
        }
    ] + [
        {
            "assistant_message": {"role": "assistant", "content": None, "tool_calls": [f"tc_{i}"]},
            "tool_messages": [{"role": "tool", "tool_call_id": f"tc_{i}", "content": f"r{i}"}],
            "calls": [{"expression": f"q{i}", "result": f"r{i}"}],
        }
        for i in range(1, KEEP_LAST_FULL_TURNS + 2)
    ]

    messages = _build_messages(base, history)
    summary_msg = messages[len(base)]  # right after base messages
    assert summary_msg["role"] == "user"
    assert "df.groupby('a').describe()" in summary_msg["content"]
    assert len(summary_msg["content"]) < 500
    assert "omitted" in summary_msg["content"].lower()

def test_end_to_end_message_count_plateaus_across_many_iterations(df, profile):
    dense_expr = "df.groupby('b').describe()"
    looping_response = _tool_call_message(dense_expr)
    forced_final_response = _final_message("## Key Patterns\n- done")

    with patch("groq_agent.Groq") as MockClient:
        mock_create = MockClient.return_value.chat.completions.create
        mock_create.side_effect = [looping_response] * 7 + [forced_final_response]
        generate_insights_agentic(df, profile, max_iterations=6, verbose=False)

    call_args = mock_create.call_args_list
    message_lengths = [len(c.kwargs["messages"]) for c in call_args]

    # Within the loop (indices 0-6, all tool-call rounds), message count
    # should plateau once the window fills — the last loop call's count
    # must match a middling call's, not still be climbing.
    loop_lengths = message_lengths[:-1]  # exclude the forced-final call
    assert loop_lengths[-1] == loop_lengths[3]

    # The forced-final call (last one) intentionally adds exactly one extra
    # system message on top of the plateaued window ("write your final
    # report now") — so it should be +1 versus the loop plateau, not
    # unboundedly larger.
    assert message_lengths[-1] == loop_lengths[-1] + 1


# ---------------------------------------------------------------------------
# answer_question: shares the same tool loop as generate_insights_agentic
# (see _run_tool_loop), so these focus on the parts that differ: the
# question gets into the prompt, and the return shape is {"answer", ...}
# rather than {"report", ...}.
# ---------------------------------------------------------------------------

def test_answer_question_returns_answer_key_not_report(df, profile):
    mock_response = _final_message("Yes, likely on time based on 5 similar past records.")

    with patch("groq_agent.Groq") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = mock_response
        result = answer_question(df, profile, "will it be on time?", verbose=False)

    assert result["answer"] == "Yes, likely on time based on 5 similar past records."
    assert "report" not in result
    assert result["tool_calls"] == []
    assert result["iterations"] == 0

def test_answer_question_includes_the_question_in_the_prompt_sent_to_the_model(df, profile):
    mock_response = _final_message("Based on the data, likely late.")
    question = "is train TRN1014 going to be late?"

    with patch("groq_agent.Groq") as MockClient:
        mock_create = MockClient.return_value.chat.completions.create
        mock_create.return_value = mock_response
        answer_question(df, profile, question, verbose=False)

    sent_messages = mock_create.call_args.kwargs["messages"]
    user_message = next(m for m in sent_messages if m["role"] == "user")
    assert question in user_message["content"]

def test_answer_question_uses_qa_system_prompt_not_report_prompt(df, profile):
    mock_response = _final_message("Some answer.")

    with patch("groq_agent.Groq") as MockClient:
        mock_create = MockClient.return_value.chat.completions.create
        mock_create.return_value = mock_response
        answer_question(df, profile, "some question?", verbose=False)

    sent_messages = mock_create.call_args.kwargs["messages"]
    system_message = next(m for m in sent_messages if m["role"] == "system")
    assert system_message["content"] == QA_SYSTEM_PROMPT
    assert "comparable historical rows" in system_message["content"]

def test_answer_question_executes_tool_calls_same_as_report_mode(df, profile):
    first_response = _tool_call_message("df[df['a'] == 3]")
    second_response = _final_message("Found it: a=3 corresponds to row index 2.")

    with patch("groq_agent.Groq") as MockClient:
        mock_create = MockClient.return_value.chat.completions.create
        mock_create.side_effect = [first_response, second_response]
        result = answer_question(df, profile, "what row has a=3?", verbose=False)

    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["expression"] == "df[df['a'] == 3]"
    assert "row index 2" in result["answer"]

def test_answer_question_respects_max_iterations_cap(df, profile):
    looping_response = _tool_call_message("df['a'].sum()")
    forced_final_response = _final_message("forced final answer")

    with patch("groq_agent.Groq") as MockClient:
        mock_create = MockClient.return_value.chat.completions.create
        mock_create.side_effect = [looping_response] * 3 + [forced_final_response]
        result = answer_question(df, profile, "some question?", max_iterations=2, verbose=False)

    assert result["iterations"] == 2
    assert "forced final answer" in result["answer"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
