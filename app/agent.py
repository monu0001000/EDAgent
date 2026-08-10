"""
agent.py
Phase 3b: Agentic insight generator — using Google Gemini's free API tier.

Upgrade over insight_generator.py's single-shot approach: instead of only
seeing pre-computed profile stats, the LLM is given a `run_pandas_query`
tool (backed by sandbox.safe_query) and can decide, turn by turn, what
follow-up questions to investigate — e.g. "the profile shows a strong
correlation between spend and tickets, let me check if that holds within
each plan tier separately" — before writing its final report.

This uses Gemini's manual (non-automatic) function-calling flow: we send
the tool declaration ourselves, inspect response.function_calls, execute
each one through the sandbox, and feed the results back as a `tool` role
turn. We deliberately do NOT use the SDK's automatic-function-calling
convenience (which would call a real Python function directly) because we
want every query to go through the sandbox's AST safety checks first.

Setup: get a free API key at https://aistudio.google.com/apikey (no credit
card required) and set it as GEMINI_API_KEY in your .env file.
"""

import os
import time
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors

load_dotenv()  # reads .env in the current/parent directory automatically —
                # means GEMINI_API_KEY doesn't need to be manually exported

from sandbox import safe_query, UnsafeQueryError


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
Missing data, outliers, duplicates, or PII columns needing attention. Say briefly if none are notable.

## Suggested Questions
3-4 business questions this dataset could help answer.

Rules:
- Never invent statistics — only cite numbers from the profile or from your own tool query results.
- Be concise: short bullets, not paragraphs.
- Flag any PII columns as needing masking before wider distribution.
"""

RUN_PANDAS_QUERY_DECLARATION = types.FunctionDeclaration(
    name="run_pandas_query",
    description=(
        "Run a single read-only pandas expression against the dataframe (available as `df`). "
        "Returns a string of the result, truncated if large. No assignments, imports, or "
        "arbitrary code execution are allowed — expression only, e.g. "
        "\"df.groupby('plan')['spend'].mean()\"."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "expression": types.Schema(
                type=types.Type.STRING,
                description="A single pandas expression using `df`, e.g. \"df.groupby('plan')['spend'].corr(df['tickets'])\"",
            )
        },
        required=["expression"],
    ),
)

TOOL = types.Tool(function_declarations=[RUN_PANDAS_QUERY_DECLARATION])


def _generate_with_retry(client, model, contents, config, max_retries: int = 4, verbose: bool = True):
    """
    Wraps client.models.generate_content with retry-with-backoff specifically
    for 429 RESOURCE_EXHAUSTED errors. The free tier's rate limit (as low as
    5 requests/minute on some models) is easily hit by this agent's loop,
    which can make several calls in quick succession — so treat 429 as a
    "wait and retry" case rather than letting it bubble up as a hard failure.
    """
    delay = 15  # seconds; Gemini's 429 responses typically suggest ~20-30s
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except errors.ClientError as e:
            if e.code == 429 and attempt < max_retries - 1:
                if verbose:
                    print(f"[agent] rate limited (429), waiting {delay}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(delay)
                delay = min(delay * 1.5, 60)
                continue
            raise


def _build_initial_prompt(profile: dict) -> str:
    import json
    return (
        "Here is the structured profile of the dataset:\n\n"
        f"{json.dumps(profile, indent=2, default=str)}\n\n"
        "Investigate anything worth digging into using run_pandas_query, then write your final report."
    )


def generate_insights_agentic(
    df: pd.DataFrame,
    profile: dict,
    model: str | None = None,
    max_iterations: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Runs the agentic loop. Returns a dict with:
      - "report": final markdown text
      - "tool_calls": list of {expression, result} the agent actually ran (for transparency/logging)
      - "iterations": how many tool-use rounds happened

    Model defaults to the GEMINI_MODEL env var if set, otherwise "gemini-flash-latest" — a
    Google-maintained alias that always points at their current recommended free Flash
    model, so it won't break every time they retire/rename a specific version (as
    gemini-2.5-flash and gemini-3-flash both did during this project). If you ever want a
    pinned version instead, run list_models.py to see everything your key can access, then
    set GEMINI_MODEL in .env.
    """
    model = model or os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=_build_initial_prompt(profile))])
    ]
    tool_call_log = []

    config = types.GenerateContentConfig(
        system_instruction=AGENT_SYSTEM_PROMPT,
        tools=[TOOL],
        max_output_tokens=3000,
    )

    for iteration in range(max_iterations + 1):  # +1 to allow a final non-tool response
        response = _generate_with_retry(client, model, contents, config, verbose=verbose)

        function_calls = response.function_calls or []

        if not function_calls:
            return {"report": response.text, "tool_calls": tool_call_log, "iterations": iteration}

        # Append the model's own turn (the function_call parts) to the conversation
        contents.append(response.candidates[0].content)

        # Execute each requested tool call through the sandbox, build function_response parts
        response_parts = []
        for fc in function_calls:
            expression = fc.args.get("expression", "")
            if verbose:
                print(f"[agent] running query: {expression}")
            try:
                result = safe_query(df, expression)
            except UnsafeQueryError as e:
                result = f"Query rejected for safety reasons: {e}"

            tool_call_log.append({"expression": expression, "result": result})
            response_parts.append(
                types.Part.from_function_response(name=fc.name, response={"result": result})
            )

        # Note: some Gemini docs show role="tool" for function responses, but
        # the live API currently only accepts USER/MODEL roles for Content —
        # role="tool" gets rejected with a 400 INVALID_ARGUMENT. Using "user".
        contents.append(types.Content(role="user", parts=response_parts))

    # Hit max_iterations without a final answer — force one more call without
    # tools so we still return something useful instead of silently failing.
    forced_config = types.GenerateContentConfig(
        system_instruction=AGENT_SYSTEM_PROMPT
        + "\n\nYou have used all your tool calls. Write your final report now based on what you've learned.",
        max_output_tokens=3000,
    )
    response = _generate_with_retry(client, model, contents, forced_config, verbose=verbose)
    return {"report": response.text, "tool_calls": tool_call_log, "iterations": max_iterations}


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
    print("Running agentic insight generation...\n")
    result = generate_insights_agentic(test_df, profile, verbose=True)
    print("\n=== TOOL CALLS MADE ===")
    for tc in result["tool_calls"]:
        print(f"  {tc['expression']}  ->  {tc['result'][:100]}")
    print(f"\n=== FINAL REPORT (after {result['iterations']} tool-use rounds) ===\n")
    print(result["report"])
