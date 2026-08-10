# EDAgent

An AI-powered data analysis agent: upload any CSV and it automatically profiles the data,
generates relevant visualizations, and uses an LLM (with tool-calling) to investigate
patterns and write a natural-language insights report — deciding what to dig into rather
than just summarizing fixed statistics.

Runs entirely on **Google Gemini's free API tier** (no credit card, no cost) — see Setup below.

## Project status

| Component | File | Status |
|---|---|---|
| Data profiling | `app/profiler.py` | ✅ Done, tested |
| Auto-visualization | `app/visualizer.py` | ✅ Done, tested |
| Sandboxed query executor | `app/sandbox.py` | ✅ Done, tested (12 escape-attempt regression tests) |
| Single-shot LLM summarizer | `app/insight_generator.py` | ✅ Done, confirmed working live |
| Agentic insight generator | `app/agent.py` | ✅ Done, confirmed working live |
| Streamlit UI | `app/streamlit_app.py` | ✅ Done, tested |

**46/46 tests passing** across `test_pipeline.py`, `test_agent.py`, `test_streamlit_app.py`.

## Setup

1. Get a **free** API key at https://aistudio.google.com/apikey — no credit card required.
2. Install dependencies and set the key:

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env and paste your real GEMINI_API_KEY
```

`.env` is loaded automatically (via `python-dotenv`) — no manual `export`/`set` needed.

Note: the free tier currently covers Flash-class models, not the Pro-series models, and
Google may use free-tier prompts to improve their products — fine for a portfolio project,
just don't feed it real sensitive data. The model defaults to `gemini-flash-latest`, a
Google-maintained alias for their current recommended free Flash model — if that ever
breaks, run `python app/list_models.py` to see exactly what your key can access and set
`GEMINI_MODEL` in `.env` to pin a specific one.

## Run the app

```bash
cd app
streamlit run streamlit_app.py
```

Upload a CSV, review the auto-generated profile and charts, then pick a report mode:
- **Agentic** — the model decides what's worth investigating and runs its own pandas
  queries (via the sandboxed tool) before writing the report. Slower, sharper.
- **Single-shot** — one call, summarizes the pre-computed profile only. Faster, simpler.

## Run individual pieces / tests

```bash
cd app
python3 profiler.py           # profiler smoke test, no API key needed
python3 visualizer.py         # chart generation smoke test, no API key needed
python3 sandbox.py            # sandbox security smoke test, no API key needed
python3 list_models.py        # see which models your API key can actually use
python3 insight_generator.py  # single-shot report — needs GEMINI_API_KEY
python3 agent.py              # agentic report — needs GEMINI_API_KEY, watch it investigate

python3 -m pytest -v          # full test suite (46 tests)
```

## Architecture

```
CSV → profiler.py (structured profile: types, missing, outliers, correlations)
        ↓
      visualizer.py (auto-generated plotly charts based on column types)
        ↓
      agent.py (Gemini investigates via sandboxed run_pandas_query tool, then writes report)
        ↓
      streamlit_app.py (upload → profile/charts → report, in the browser)
```

Design choices worth noting:

- **The LLM never sees raw rows** — only the aggregated profile dict (column stats, not
  data). This keeps token usage flat regardless of dataset size (~700 tokens for a
  7-column dataset, scales with column count not row count) and avoids leaking sensitive
  values unnecessarily.
- **The agent's query tool is sandboxed via AST inspection, not `exec()`.** It parses each
  expression, statically rejects imports/assignments/dunder access/`.format()`-based
  bypasses, then evaluates with an empty `__builtins__`. See `sandbox.py` docstring for
  the specific bypass (`"{0.__class__}".format(df)`) that was found and patched during
  development — string-format-based dunder access doesn't show up as an `Attribute` node
  in the AST, so a naive "block dunder attribute access" check alone isn't sufficient.
- **PII detection is type-aware** to avoid false positives (e.g. a datetime column like
  `"2023-01-01"` structurally resembles a phone-number regex; the check skips
  pattern-matching for numeric/datetime columns and relies on column-name hints instead).
- **Manual (not automatic) function calling.** The Gemini SDK can auto-execute Python
  functions passed as tools, but we deliberately bypass that convenience and handle each
  tool call ourselves so every query goes through the sandbox's AST safety checks before
  it ever touches the real dataframe.
- **Provider-agnostic core.** The agent loop (`agent.py`) was originally built against the
  Anthropic API and migrated to Gemini with no changes to `profiler.py`, `visualizer.py`,
  or `sandbox.py` — only the API client and message-formatting code changed.
- **UI tested without a browser.** `test_streamlit_app.py` uses Streamlit's `AppTest`
  framework to drive real file uploads through the app headlessly, catching integration
  bugs (caching, render pipeline) that unit tests on the individual modules alone can't.

## Next steps

- PDF/HTML export of the final report
- Evaluation rubric: run the agent across N test datasets, score insight relevance/accuracy
- Deploy publicly (Streamlit Community Cloud or Hugging Face Spaces) for a live demo link

## Resume bullets

- Built EDAgent, an AI-powered data analysis tool that autonomously profiles, visualizes,
  and generates natural-language insight reports from arbitrary CSV files, using LLM
  tool-calling to dynamically drive exploratory analysis rather than a fixed pipeline.
- Designed and hardened a sandboxed pandas-query execution layer (AST-based static
  analysis, not `exec()`) allowing an LLM agent to safely query dataframes iteratively;
  identified and patched a string-format-based sandbox escape during development, with
  regression tests covering 12 distinct bypass attempts.
- Achieved flat LLM token usage regardless of dataset size by sending only aggregated
  profile statistics, never raw rows, to the model.
- Decoupled the agent's core logic from a specific LLM provider, migrating the tool-calling
  loop from Anthropic's API to Google Gemini's free tier with no changes to the data
  profiling, visualization, or sandboxing layers.
- Shipped a Streamlit UI with headless integration tests (Streamlit `AppTest`) covering the
  full upload-to-report flow, reaching 46 passing tests across the project.


