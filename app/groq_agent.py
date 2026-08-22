"""
groq_agent.py
Agentic insight generator using Groq's free API tier: the LLM decides what's
worth investigating, runs its own pandas queries against the dataframe (via
the sandboxed run_pandas_query tool), and iterates before writing a final
markdown report. Slower and more expensive than groq_insight_generator.py's
single-shot mode, but sharper — it can chase down a hypothesis instead of
only summarizing the pre-computed profile.

Groq exposes an OpenAI-compatible chat completions API: tools as JSON
schemas, tool_calls in the response, role="tool" for results.

Setup: get a free API key at https://console.groq.com/keys (no credit card
required) and set it as GROQ_API_KEY in your .env file.
"""

import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from groq import Groq, RateLimitError, BadRequestError

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
Do not use it just to re-fetch numbers already present in the profile. `.corr()` and other numeric
pandas methods only work on numeric columns — check a column's `inferred_type` in the profile before
querying it that way; if you want to relate a categorical column to a numeric one, use `.groupby()`
instead. Call the tool at most 7 times, then write your final report. If a query returns an error,
don't spend another call repeating the same mistake — either fix the specific problem (e.g. group by
the categorical column instead of correlating it directly) or move on to a different question with
your remaining calls.

Before citing any groupby/aggregate result as a finding, sanity-check it against the profile: if the
group it's based on is very small (e.g. a category with a count of 1 in `top_values`), or the
aggregate is close to a value the profile already flagged as an outlier or a `category_normalization_issues`
variant, say so explicitly rather than reporting it as a reliable pattern — a single-row average is
not a trend. This matters most exactly when a number looks the most dramatic: a delay of "999 minutes"
or similar sentinel-shaped value is far more likely to be a data-entry error than a real observation,
and reporting it uncritically is worse than not reporting it at all.

When you are done investigating, respond with your final analysis as markdown with these sections:

## Key Patterns
3-5 bullet points on the most interesting findings, citing specific numbers. If a finding came from
a tool query rather than the initial profile, make that clear (e.g. "Breaking this down by plan tier
shows..."). For each bullet, briefly note why it matters, not just what the number is.

## Anomalies Worth a Second Look
Any specific number that looks suspicious rather than trustworthy — a value far outside the rest of
the distribution, an aggregate driven by a single row, a group whose name is one of the
`category_normalization_issues` variants of another group. Explain *why* it looks suspicious (e.g.
"this group has only 1 row" or "this exact value looks like a placeholder/sentinel rather than a
measurement"). Say briefly if nothing stood out beyond what's already in Data Quality Issues.

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
- Be concise: short bullets, not paragraphs. Concise means no filler words, not fewer findings —
  if the data supports 5 distinct findings, give 5, don't compress them into fewer bullets to save space.
- Flag any PII columns as needing masking before wider distribution.
"""

QA_SYSTEM_PROMPT = """You are a data analyst answering one specific question about a dataset, asked
by someone who has the structured profile below and a run_pandas_query tool for running read-only
pandas expressions against the real dataframe (`df`).

The dataset could be about anything — transactions, customers, trains, sensors, sales, patients,
whatever the columns describe. Don't assume a domain; read the column names and the question to work
out what's actually being asked, and answer using the columns that are actually present.

Many questions ask you to make a call about ONE specific row or entity — "is train TRN1014 going to
be late?", "will customer C-8842 churn?", "is transaction TXN-991 fraudulent?" are the same underlying
task in every case. You cannot see the future and this dataset has no ground truth for what hasn't
happened yet, so answer by finding the most relevant HISTORICAL evidence and reasoning from it, not by
inventing a number:
1. First, try to find the specific row(s) the question refers to — usually by filtering `df` on an ID
   or name column.
2. Then find comparable historical rows: same route/category/segment/group, similar conditions — and
   look at the actual outcome distribution for those rows (e.g. groupby + mean/median/value_counts).
3. Base your answer on that comparison, and state how many historical rows it's based on. If it's a
   small number (say, under ~10), say so explicitly and hedge the answer accordingly — a pattern from
   3 rows is a hint, not a prediction, and this dataset may only have a handful of rows total.
4. If there's no reasonable historical comparison to make — the entity is completely novel, the ID
   isn't in the data, or there's no relevant column to compare against — say that plainly instead of
   guessing.

Rules:
- Never invent numbers — only cite figures from the profile or your own tool query results.
- If a groupby/filter result relies on a value flagged elsewhere as a likely data-entry error (e.g. it
  matches a `category_normalization_issues` variant, or looks like a sentinel/placeholder value), don't
  base the answer on it without saying so — a single-row group is not a trend.
- Structure: a direct answer first (1-2 sentences), then the evidence that supports it, then an
  explicit confidence/caveat line (e.g. "based on 6 similar past records" or "low confidence: only 2
  comparable rows exist").
- Keep the whole answer under ~150 words. This is a direct answer to a question, not a full report.
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

# Extra rounds, after max_iterations is exhausted, where the model is
# strongly told to stop investigating and answer — but tools stay
# technically available so a tool call it insists on making anyway just
# succeeds normally instead of triggering the tool_use_failed error that
# happens when tools are stripped to None (see _parse_forced_tool_call_
# from_error). Only after these also produce another tool call do we fall
# back to the harder tools=None cutoff.
WRAP_UP_ROUNDS = 2
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


def _build_question_prompt(profile: dict, question: str) -> str:
    return (
        "Here is the structured profile of the dataset:\n\n"
        f"{json.dumps(profile, indent=2, default=str)}\n\n"
        f'The user is asking: "{question}"\n\n'
        "Use run_pandas_query to find the specific row(s) this question is about (if any) and "
        "comparable historical rows, then answer."
    )


def _create_with_retry(client, model, messages, tools, max_retries: int = 4, verbose: bool = True, **kwargs):
    """Retry-with-backoff wrapper for Groq's RateLimitError. Respects the
    Retry-After response header when present, otherwise falls back to a
    fixed delay.

    Defaults reasoning_effort to "low" specifically for gpt-oss models
    (unless already set via kwargs): they're reasoning models that spend
    real time "thinking" before answering even for a straightforward
    "which groupby should I run next" decision, and with up to
    max_iterations + WRAP_UP_ROUNDS calls in a single investigation, that
    adds up — a real deployed run against a 7-column sales dataset visibly
    dragged. Gated to gpt-oss specifically (rather than applied
    unconditionally) since reasoning_effort isn't a universally-supported
    parameter — passing it to a model that doesn't recognize it risks a
    400 on any GROQ_MODEL override that isn't a gpt-oss variant."""
    if "gpt-oss" in model and "reasoning_effort" not in kwargs:
        kwargs["reasoning_effort"] = "low"

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


def _execute_tool_calls(df: pd.DataFrame, tool_calls, tool_call_log: list, verbose: bool) -> tuple[list, list]:
    """Run each requested tool call through the sandbox, logging every one
    to tool_call_log (for the investigation-log UI) and returning both the
    role="tool" messages to append to the conversation AND the plain
    {expression, result} records _compact_summary_message needs later.
    Shared by the main loop and the forced-final-answer grace handling
    below, so a tool call is always executed and logged the same way
    regardless of which branch triggered it."""
    tool_messages = []
    calls = []
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
        calls.append({"expression": expression, "result": result})
        tool_messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": result,
        })
    return tool_messages, calls


def _parse_forced_tool_call_from_error(error: BadRequestError):
    """gpt-oss-120b (and possibly other reasoning models) occasionally
    tries to call a tool even on a request where tools have been removed
    specifically to force a final text answer — Groq's API then rejects
    the whole response with a 400 tool_use_failed error rather than just
    ignoring the attempted call. The tool call the model wanted to make is
    still recoverable from the error body's `failed_generation` field, so
    rather than surface a raw 400 to the user, we can execute that one
    query for real, feed the result back, and ask again — the model
    almost always accepts the answer at that point since the thing it
    wanted to check has now actually been checked.

    Returns a tool_calls-shaped list (matching what message.tool_calls
    normally looks like) if the error is this specific, recoverable shape,
    else None."""
    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        return None
    err = body.get("error", {})
    if err.get("code") != "tool_use_failed":
        return None
    raw = err.get("failed_generation")
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if parsed.get("name") != "run_pandas_query":
        return None

    class _RecoveredToolCall:
        """Minimal stand-in for the SDK's tool_call object — just enough
        shape (`.id`, `.function.name`, `.function.arguments`) for
        _execute_tool_calls and the message-history bookkeeping to treat
        it exactly like a real one."""
        def __init__(self, arguments: str):
            self.id = "recovered_call_1"
            self.function = type("_F", (), {"name": "run_pandas_query", "arguments": arguments})()

    args = parsed.get("arguments", {})
    return [_RecoveredToolCall(json.dumps(args) if not isinstance(args, str) else args)]


def _run_tool_loop(
    df: pd.DataFrame,
    system_prompt: str,
    initial_user_prompt: str,
    model: str | None,
    max_iterations: int,
    verbose: bool,
) -> dict:
    """The actual agentic tool-calling loop, shared by every task that
    needs "let the model run its own pandas queries against `df` before
    answering" — currently generate_insights_agentic (writes a full
    report) and answer_question (answers one specific question). Only the
    system prompt and initial user message differ between tasks; the
    message-history bookkeeping, tool execution, retry/backoff, and
    forced-final-answer fallback are identical either way, so they live
    here once instead of being copy-pasted per task.

    Returns {"final_text": str, "tool_calls": [...], "iterations": int}.
    """
    model = model or os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
    # max_retries=0 is deliberate: the SDK has its own built-in retry-
    # with-backoff, separate from and invisible to _create_with_retry
    # below. Both retrying independently means a rate-limited request can
    # sleep for a long, unpredictable, unlogged stretch inside the SDK's
    # own retry before our code (or gunicorn's request timeout) ever gets
    # a chance to see what's happening — this is exactly what crashed a
    # real deployment: gunicorn's worker timeout fired mid-sleep inside
    # the SDK's internal retry and got SIGKILLed, with none of our own
    # retry logging ever printed. Disabling the SDK's retries puts all
    # retry behavior through _create_with_retry instead, which is
    # visible (prints each attempt), respects Retry-After, and is capped.
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"), timeout=45.0, max_retries=0)

    base_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_user_prompt},
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
            return {"final_text": message.content, "tool_calls": tool_call_log, "iterations": iteration}

        tool_messages, calls = _execute_tool_calls(df, tool_calls, tool_call_log, verbose)
        iteration_history.append({
            "assistant_message": message,
            "tool_messages": tool_messages,
            "calls": calls,
        })

    # Ran out of iterations. First, give the model up to WRAP_UP_ROUNDS
    # extra chances to answer in plain text while tools STAY available
    # (just with a strong "stop investigating, answer now" instruction).
    # This is deliberately different from immediately stripping tools to
    # None: gpt-oss-120b doesn't always respect a hard "no tools" removal
    # (see _parse_forced_tool_call_from_error below) and gets a real 400
    # error when it tries anyway — but if tools are still technically
    # available, a tool call it insists on making just succeeds normally,
    # bounded to a couple of extra rounds rather than crashing outright.
    # This fixed a real case where investigating a 7-column sales dataset
    # ran the model out of patience for one more round of synthesis and it
    # kept reaching for the tool instead of writing the report.
    for _wrap_up_round in range(WRAP_UP_ROUNDS):
        messages = _build_messages(base_messages, iteration_history)
        messages.append({
            "role": "system",
            "content": (
                "You have used your full investigation budget. Do not call any more tools — "
                "write your final answer now as plain text, based on everything you've already found."
            ),
        })
        response = _create_with_retry(
            client, model, messages, TOOLS,
            max_completion_tokens=3000, temperature=0.3, verbose=verbose,
        )
        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        if not tool_calls:
            return {"final_text": message.content, "tool_calls": tool_call_log, "iterations": max_iterations}

        tool_messages, calls = _execute_tool_calls(df, tool_calls, tool_call_log, verbose)
        iteration_history.append({
            "assistant_message": message,
            "tool_messages": tool_messages,
            "calls": calls,
        })

    # Absolute last resort: strip tools entirely. Tools are deliberately
    # omitted (tools=None) so the model can't just keep investigating
    # forever — but gpt-oss-120b sometimes tries to call one anyway
    # despite that, which Groq's API rejects outright as a 400 rather than
    # silently ignoring. Recover from that specific case once: actually
    # run the query it wanted, feed the result back, then ask again. If it
    # insists a second time, give up gracefully with whatever's already
    # been found rather than raising.
    final_messages = _build_messages(base_messages, iteration_history)
    final_messages.append({
        "role": "system",
        "content": "You have used all your tool calls. Give your final answer now based on what you've learned.",
    })

    for grace_attempt in range(2):
        try:
            response = _create_with_retry(
                client, model, final_messages, None, max_completion_tokens=3000, temperature=0.3, verbose=verbose,
            )
            return {
                "final_text": response.choices[0].message.content,
                "tool_calls": tool_call_log,
                "iterations": max_iterations,
            }
        except BadRequestError as e:
            recovered_tool_calls = _parse_forced_tool_call_from_error(e)
            if recovered_tool_calls is None or grace_attempt == 1:
                break  # not the recoverable case, or already tried once — stop retrying
            if verbose:
                print("[agent] model tried one more tool call while being forced to finish — running it, then asking again")
            tool_messages, _calls = _execute_tool_calls(df, recovered_tool_calls, tool_call_log, verbose)
            # The model's own "assistant" turn that requested this call was
            # never actually returned to us (the request that contained it
            # got rejected), so there's no real assistant message to log
            # here — just the tool result, appended directly as context for
            # the next attempt.
            final_messages.extend(tool_messages)

    # Every attempt failed (or the error wasn't the recoverable shape) —
    # rather than crash the whole report/answer on what the model's own
    # investigation already turned up, summarize from tool_call_log.
    if tool_call_log:
        fallback_lines = "\n".join(f"- `{tc['expression']}` → {tc['result'][:200]}" for tc in tool_call_log)
        fallback_text = (
            "The model couldn't produce a final written answer after its investigation "
            f"(hit an API error while wrapping up), but here's what it found:\n\n{fallback_lines}"
        )
    else:
        fallback_text = "The model couldn't produce an answer due to a repeated API error, and hadn't run any queries yet."
    return {"final_text": fallback_text, "tool_calls": tool_call_log, "iterations": max_iterations}


def generate_insights_agentic(
    df: pd.DataFrame,
    profile: dict,
    model: str | None = None,
    max_iterations: int = 7,
    verbose: bool = True,
) -> dict:
    """
    Same interface and return shape as agent.generate_insights_agentic, but
    using Groq instead of Gemini. Model defaults to the GROQ_MODEL env var
    if set, otherwise "openai/gpt-oss-120b" — OpenAI's open-weight MoE model
    served on Groq's free tier, with tool-calling support. Migrated from
    llama-3.3-70b-versatile after Groq deprecated it (Aug 16, 2026); see
    README's model migration note if this needs to change again.

    max_iterations default raised from 5 to 7: with 5, a single failed
    query (e.g. calling .corr() on a non-numeric column) could burn the
    agent's last call and force it to write the final report having never
    recovered from the mistake — observed in practice on a real messy
    dataset, see README.
    """
    result = _run_tool_loop(
        df, AGENT_SYSTEM_PROMPT, _build_initial_prompt(profile),
        model, max_iterations, verbose,
    )
    return {
        "report": result["final_text"],
        "tool_calls": result["tool_calls"],
        "iterations": result["iterations"],
    }


def answer_question(
    df: pd.DataFrame,
    profile: dict,
    question: str,
    model: str | None = None,
    max_iterations: int = 6,
    verbose: bool = True,
) -> dict:
    """
    Answer a specific natural-language question about the dataset, e.g.
    "is train TRN1014 going to be late?", "will customer C-8842 churn?",
    "is transaction TXN-991 fraudulent?" — the same underlying task in
    every case: find the specific entity, find comparable historical rows,
    reason from the outcome distribution, state confidence based on how
    much comparable history actually exists. Domain-agnostic by design —
    see QA_SYSTEM_PROMPT; nothing here assumes what the columns mean.

    Returns {"answer": str, "tool_calls": [...], "iterations": int}.
    """
    result = _run_tool_loop(
        df, QA_SYSTEM_PROMPT, _build_question_prompt(profile, question),
        model, max_iterations, verbose,
    )
    return {
        "answer": result["final_text"],
        "tool_calls": result["tool_calls"],
        "iterations": result["iterations"],
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
