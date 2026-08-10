"""
insight_generator.py
Phase 3a: Simple (single-shot) LLM summarizer — using Google Gemini's free
API tier (Google AI Studio) instead of a paid API, so this can be run and
demoed without spending money.

Takes the structured profile dict from profiler.py (NOT the raw dataframe —
we deliberately never send raw row-level data to the LLM, only aggregated
stats, to keep token usage low and avoid leaking sensitive values) plus the
list of chart names from visualizer.py, and asks Gemini to write a natural
-language analysis: key patterns, data quality issues, and suggested
business questions.

This is intentionally single-shot (one prompt -> one response) as a
baseline. agent.py upgrades this to an agentic loop where the LLM can
request additional pandas queries instead of only seeing pre-computed stats.

Setup: get a free API key at https://aistudio.google.com/apikey (no credit
card required) and set it as GEMINI_API_KEY in your .env file.
"""

import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()  # reads .env in the current/parent directory automatically —
                # means GEMINI_API_KEY doesn't need to be manually exported


SYSTEM_PROMPT = """You are a senior data analyst reviewing a dataset profile prepared by an automated tool.
You will be given:
1. A structured JSON profile: shape, column types, missing values, outliers, PII flags, and correlations.
2. A list of chart names that were auto-generated from this data (you don't see the images, just names).

Write a concise analysis with three sections, using markdown headers:

## Key Patterns
3-5 bullet points on the most interesting/important findings (correlations, distributions, category imbalances, etc.)

## Data Quality Issues
Bullet points on missing data, outliers, duplicates, or PII columns that need attention before this data is used
for modeling or reporting. If there are no notable issues, say so briefly.

## Suggested Questions
3-4 business questions this dataset could help answer, based on what's actually present in the columns.

Rules:
- Only reference columns and numbers that actually appear in the profile. Never invent statistics.
- Be specific: name columns and cite the actual numbers from the profile (e.g. "age has 6 missing values (3.0%)").
- Keep it concise — this is a report section, not an essay. Use short bullets, not long paragraphs.
- If a column is flagged as PII, mention it should be masked/excluded before wider distribution.
"""


def build_user_prompt(profile: dict, chart_names: list[str]) -> str:
    """Build the user-turn content sent to the LLM. Separated out so it can be
    unit-tested / inspected without needing to make a live API call."""
    return (
        "Dataset profile:\n"
        f"{json.dumps(profile, indent=2, default=str)}\n\n"
        "Auto-generated chart names (for context on what visuals accompany this report):\n"
        f"{json.dumps(chart_names, indent=2)}"
    )


def generate_insights(profile: dict, chart_names: list[str], model: str | None = None) -> str:
    """Call the Gemini API once with the profile + chart list, return markdown insights text.

    Model defaults to the GEMINI_MODEL env var if set, otherwise "gemini-flash-latest" — a
    Google-maintained alias that always points at their current recommended free Flash
    model, so it won't break every time they retire/rename a specific version. Run
    list_models.py to see everything your key can access if you want a pinned version.
    """
    model = model or os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    user_content = build_user_prompt(profile, chart_names)

    response = client.models.generate_content(
        model=model,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=3000,  # newer models can spend part of this on internal
                                      # reasoning before the visible answer, so give headroom
        ),
    )

    return response.text


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from profiler import profile_dataframe
    from visualizer import generate_charts

    rng = np.random.default_rng(0)
    n = 200
    monthly_spend = rng.normal(50, 15, n)
    monthly_spend[0] = 5000  # inject an outlier
    test_df = pd.DataFrame({
        "customer_id": range(1, n + 1),
        "email": [f"user{i}@test.com" for i in range(n)],
        "age": rng.integers(18, 90, n).astype(float),
        "signup_date": pd.date_range("2023-01-01", periods=n, freq="D").astype(str),
        "plan": rng.choice(["basic", "pro", "enterprise"], n, p=[0.7, 0.2, 0.1]),
        "monthly_spend": monthly_spend,
        "support_tickets": (monthly_spend / 10 + rng.normal(0, 1, n)),
    })
    test_df.loc[5:10, "age"] = np.nan

    profile = profile_dataframe(test_df)
    charts = generate_charts(test_df, profile)

    print("Calling Gemini API for insights...\n")
    insights = generate_insights(profile, list(charts.keys()))
    print(insights)
