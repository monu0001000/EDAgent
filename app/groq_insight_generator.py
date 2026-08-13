"""
groq_insight_generator.py
Single-shot LLM summarizer using Groq's free API tier — the Groq equivalent
of insight_generator.py (which uses Gemini). Same design: one call,
summarizes the pre-computed profile only, no tool use.

Setup: get a free API key at https://console.groq.com/keys (no credit card
required) and set it as GROQ_API_KEY in your .env file.
"""

import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

SYSTEM_PROMPT = """You are a senior data analyst reviewing a dataset profile prepared by an automated tool.
You will be given:
1. A structured JSON profile: shape, column types, missing values, outliers, PII flags, and correlations.
2. A list of chart names that were auto-generated from this data (you don't see the images, just names).

Write a concise analysis with three sections, using markdown headers:

## Key Patterns
3-5 bullet points on the most interesting/important findings (correlations, distributions, category imbalances, etc.)

## Data Quality Issues
Bullet points on missing data, outliers, duplicates, or PII columns that need attention before this data is used
for modeling or reporting. Specifically check each column's profile for these fields, which flag issues code
already detected deterministically — surface them explicitly if present, don't just paraphrase the raw JSON:
- `category_normalization_issues`: values that are likely the same category but differ by whitespace/case
  (e.g. "Bandra" vs "  Bandra  ", "High" vs "high") — these should probably be merged before analysis.
- `duplicate_id_values`: a column that looks like it should be a unique identifier but has repeated values.
- `mixed_numeric_text`: a column that's mostly numeric but has some non-numeric values mixed in.
- `unparseable_values`: a datetime column with values present but invalid (e.g. an impossible time like "25:00").
If there are no notable issues, say so briefly.

## Suggested Questions
3-4 business questions this dataset could help answer, based on what's actually present in the columns.

Rules:
- Only reference columns and numbers that actually appear in the profile. Never invent statistics.
- Be specific: name columns and cite the actual numbers from the profile (e.g. "age has 6 missing values (3.0%)").
- Keep it concise — this is a report section, not an essay. Use short bullets, not long paragraphs.
- If a column is flagged as PII, mention it should be masked/excluded before wider distribution.
"""


def build_user_prompt(profile: dict, chart_names: list[str]) -> str:
    return (
        "Dataset profile:\n"
        f"{json.dumps(profile, indent=2, default=str)}\n\n"
        "Auto-generated chart names (for context on what visuals accompany this report):\n"
        f"{json.dumps(chart_names, indent=2)}"
    )


def generate_insights(profile: dict, chart_names: list[str], model: str | None = None) -> str:
    """Call Groq once with the profile + chart list, return markdown insights text.

    Model defaults to the GROQ_MODEL env var if set, otherwise "llama-3.3-70b-versatile".
    """
    model = model or os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(profile, chart_names)},
        ],
        max_completion_tokens=3000,
        temperature=0.3,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from profiler import profile_dataframe
    from visualizer import generate_charts

    rng = np.random.default_rng(0)
    n = 200
    monthly_spend = rng.normal(50, 15, n)
    monthly_spend[0] = 5000
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

    print("Calling Groq API for insights...\n")
    insights = generate_insights(profile, list(charts.keys()))
    print(insights)
