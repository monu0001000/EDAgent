"""
groq_agent.py
Agentic insight generator using Groq's free API tier as an alternative to
agent.py (which uses Gemini). Groq's free tier has a much higher per-minute
request limit (30 RPM vs as low as 5 RPM on some Gemini models), which
matters a lot for this project since the agentic loop can make several
calls in quick succession.

Groq exposes an OpenAI-compatible chat completions API, so the tool-calling
shape here (tools as JSON schemas, tool_calls in the response, role="tool"
for results) differs from agent.py's Gemini-specific Content/Part objects,
but the underlying design — sandboxed run_pandas_query tool, iterative
investigation, then a final markdown report — is identical.

Setup: get a free API key at https://console.groq.com/keys (no credit card
required) and set it as GROQ_API_KEY in your .env file.
"""

import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from groq import Groq, RateLimitError

from sandbox import safe_query, UnsafeQueryError

load_dotenv()

AGENT_SYSTEM_PROMPT = """You are a senior data analyst investigating a dataset. You have already been given
a structured profile (column types, missing values, outliers, correlations) as a starting point.

You also have a tool, run_pandas_query, that lets you run a single read-only pandas expression
against the actual dataframe (available to your expression as `df`) to investigate anything the
profile alone doesn't answer — e.g. checking whether a correlation holds within subgroups, comparing
means across categories, or spot-checking rows that triggered an outlier flag.

Use the tool when it would meaningfully sharpen your analysis — for example, breaking down a
top-level number by a categorical column, or verifying a hypothesis about *why* a pattern exists.
Do not use it just to re-fetch numbers already present in the profile. Call it at most 5 times, then
write your final report.

When you are done investigating, respond with your final analysis as markdown with these sections:

## Key Patterns
3-5 bullet points on the most interesting findings, citing specific numbers. If a finding came from
a tool query rather than the initial profile, make that clear (e.g. "Breaking this down by plan tier
shows...").

## Data Quality Issues
Missing data, outliers, duplicates, or PII columns needing attention. Specifically check each column's
profile for these fields, which flag issues code already detected deterministically — surface them
explicitly if present, don't just paraphrase the raw JSON:
- `category_normalization_issues`: values likely the same category but differing by whitespace/case
  (e.g. "Bandra" vs "  Bandra  ", "High" vs "high") — probably should be merged before analysis.
- `duplicate_id_values`: a column that looks like it should be a unique identifier but has repeats.
- `mixed_numeric_text`: a column that's mostly numeric but has some non-numeric values mixed in.
- `unparseable_values`: a datetime column with values present but invalid (e.g. an impossible "25:00").
Say briefly if none are notable.

## Suggested Questions
3-4 business questions this dataset could help answer.

Rules:
- Never invent statistics — only cite numbers from the profile or from your own tool query results.
- Be concise: short bullets, not paragraphs.
- Flag any PII columns as needing masking before wider distribution.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_pandas_query",
            "description": (
                "Run a single read-only pandas expression against the dataframe (available as `df`). "
                "Returns a string of the result, truncated if large. No assignments, imports, or "
                "arbitrary code execution are allowed — expression only, e.g. "
                "\"df.groupby('plan')['spend'].mean()\"."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A single pandas expression using `df`, e.g. \"df.groupby('plan')['spend'].corr(df['tickets'])\"",
                    }
                },
                "required": ["expression"],
            },
        },
    }
]

# How many recent tool-call iterations to keep in full in the message
# history. Older iterations get merged into a single compact summary
# instead of staying in full forever — without this, `messages` grows by
# 2+ entries every tool call (the assistant's tool_calls message + one
# "tool" role message per call), and the full text of every past result
# gets re-sent on every subsequent API call.
KEEP_LAST_FULL_TURNS = 2
SUMMARY_RESULT_CHARS = 150


def _compact_summary_message(iteration_records: list[dict]) -> dict:
    """Merge ALL older iterations into a single 'user' role message (not
    one message per iteration — that would still grow message count
    linearly), replacing full assistant tool_calls + tool result messages
    with one short digest line per past query."""
    lines = ["[Earlier investigation steps — full results omitted to save context]"]
    for record in iteration_records:
        for c in record["calls"]:
            snippet = c["result"][:SUMMARY_RESULT_CHARS]
            if len(c["result"]) > SUMMARY_RESULT_CHARS:
                snippet += "..."
            lines.append(f'Ran `{c["expression"]}` -> {snippet}')
    return {"role": "user", "content": "\n".join(lines)}


def _build_messages(base_messages: list[dict], iteration_history: list[dict]) -> list[dict]:
    """Reconstruct the full `messages` list sent to the API from scratch
    each turn: system + initial user prompt, then ONE merged compact
    summary covering all iterations older than the recent window, then the
    full assistant/tool messages for the most recent KEEP_LAST_FULL_TURNS
    iterations. Caps message count at a small constant regardless of how
    many tool calls the investigation makes."""
    messages = list(base_messages)
    older = iteration_history[:-KEEP_LAST_FULL_TURNS] if KEEP_LAST_FULL_TURNS > 0 else iteration_history
    recent = iteration_history[len(older):]

    if older:
        messages.append(_compact_summary_message(older))
    for record in recent:
        messages.append(record["assistant_message"])
        messages.extend(record["tool_messages"])

    return messages


def _build_initial_prompt(profile: dict) -> str:
    return (
        "Here is the structured profile of the dataset:\n\n"
        f"{json.dumps(profile, indent=2, default=str)}\n\n"
        "Investigate anything worth digging into using run_pandas_query, then write your final report."
    )


def _create_with_retry(client, model, messages, tools, max_retries: int = 4, verbose: bool = True, **kwargs):
    """Retry-with-backoff wrapper for Groq's RateLimitError. Respects the
    Retry-After response header when present, otherwise falls back to a
    fixed delay."""
    delay = 10
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(model=model, messages=messages, tools=tools, **kwargs)
        except RateLimitError as e:
            if attempt >= max_retries - 1:
                raise
            retry_after = None
            try:
                retry_after = float(e.response.headers.get("retry-after", ""))
            except (AttributeError, ValueError, TypeError):
                pass
            wait = retry_after if retry_after else delay
            if verbose:
                print(f"[agent] rate limited, waiting {wait:.0f}s before retry {attempt + 1}/{max_retries}...")
            time.sleep(wait)
            delay = min(delay * 1.5, 45)


def generate_insights_agentic(
    df: pd.DataFrame,
    profile: dict,
    model: str | None = None,
    max_iterations: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Same interface and return shape as agent.generate_insights_agentic, but
    using Groq instead of Gemini. Model defaults to the GROQ_MODEL env var
    if set, otherwise "llama-3.3-70b-versatile" — a strong general-purpose
    model with tool-calling support on Groq's free tier.
    """
    model = model or os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    base_messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": _build_initial_prompt(profile)},
    ]
    iteration_history = []  # list of {"assistant_message", "tool_messages", "calls"}
    tool_call_log = []

    for iteration in range(max_iterations + 1):
        messages = _build_messages(base_messages, iteration_history)
        response = _create_with_retry(
            client, model, messages, TOOLS,
            max_completion_tokens=3000, temperature=0.3, verbose=verbose,
        )
        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        if not tool_calls:
            return {"report": message.content, "tool_calls": tool_call_log, "iterations": iteration}

        tool_messages = []
        calls_this_iteration = []
        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            expression = args.get("expression", "")
            if verbose:
                print(f"[agent] running query: {expression}")
            try:
                result = safe_query(df, expression)
            except UnsafeQueryError as e:
                result = f"Query rejected for safety reasons: {e}"

            tool_call_log.append({"expression": expression, "result": result})
            calls_this_iteration.append({"expression": expression, "result": result})
            tool_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        iteration_history.append({
            "assistant_message": message,
            "tool_messages": tool_messages,
            "calls": calls_this_iteration,
        })

    # Forced final answer if we ran out of iterations
    final_messages = _build_messages(base_messages, iteration_history)
    final_messages.append({
        "role": "system",
        "content": "You have used all your tool calls. Write your final report now based on what you've learned.",
    })
    response = _create_with_retry(
        client, model, final_messages, None, max_completion_tokens=3000, temperature=0.3, verbose=verbose,
    )
    return {
        "report": response.choices[0].message.content,
        "tool_calls": tool_call_log,
        "iterations": max_iterations,
    }


if __name__ == "__main__":
    import numpy as np
    from profiler import profile_dataframe

    rng = np.random.default_rng(0)
    n = 300
    plan = rng.choice(["basic", "pro", "enterprise"], n, p=[0.6, 0.3, 0.1])
    base_spend = {"basic": 20, "pro": 60, "enterprise": 150}
    spend = np.array([base_spend[p] for p in plan]) + rng.normal(0, 10, n)
    tickets = spend / 10 + rng.normal(0, 1, n)

    test_df = pd.DataFrame({
        "customer_id": range(1, n + 1),
        "plan": plan,
        "spend": spend,
        "tickets": tickets,
        "signup_date": pd.date_range("2023-01-01", periods=n, freq="D").astype(str),
    })

    profile = profile_dataframe(test_df)
    print("Running agentic insight generation via Groq...\n")
    result = generate_insights_agentic(test_df, profile, verbose=True)
    print("\n=== TOOL CALLS MADE ===")
    for tc in result["tool_calls"]:
        print(f"  {tc['expression']}  ->  {tc['result'][:100]}")
    print(f"\n=== FINAL REPORT (after {result['iterations']} tool-use rounds) ===\n")
    print(result["report"])
