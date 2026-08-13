# EDAgent

An AI-powered data analysis agent: upload any CSV and it automatically profiles the data,
generates relevant visualizations, and uses an LLM (with tool-calling) to investigate
patterns and write a natural-language insights report — deciding what to dig into rather
than just summarizing fixed statistics.

Runs entirely on **Groq's free API tier** (no credit card, no cost) — see Setup below.

![CI](https://github.com/monu0001000/EDAgent/actions/workflows/tests.yml/badge.svg)

## Project status

| Component | File | Status |
|---|---|---|
| Data profiling | `app/profiler.py` | ✅ Done, tested |
| Auto-visualization | `app/visualizer.py` | ✅ Done, tested |
| Sandboxed query executor | `app/sandbox.py` | ✅ Done, tested (12 escape-attempt regression tests) |
| Single-shot LLM summarizer | `app/groq_insight_generator.py` | ✅ Done, tested |
| Agentic insight generator | `app/groq_agent.py` | ✅ Done, tested, rate-limit retry with backoff |
| Streamlit UI | `app/streamlit_app.py` | ✅ Done, tested |

**67/67 tests passing** across `test_pipeline.py`, `test_groq_agent.py`, `test_streamlit_app.py`.

## Setup

1. Get a **free** API key at https://console.groq.com/keys — no credit card required.
2. Install dependencies and set the key:

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env and paste your real GROQ_API_KEY
```

`.env` is loaded automatically (via `python-dotenv`) — no manual `export`/`set` needed.

Note: check Groq's current data-use policy if that matters for your use case. Fine for a
portfolio project either way, just don't feed real sensitive data through the free tier.
Defaults to `llama-3.3-70b-versatile`, overridable via `GROQ_MODEL` in `.env` if that model
ever runs out of quota or is deprecated.

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
python3 profiler.py                 # profiler smoke test, no API key needed
python3 visualizer.py               # chart generation smoke test, no API key needed
python3 sandbox.py                  # sandbox security smoke test, no API key needed
python3 groq_insight_generator.py   # single-shot report — needs GROQ_API_KEY
python3 groq_agent.py               # agentic report — needs GROQ_API_KEY, watch it investigate

python3 -m pytest -v                # full test suite (67 tests)
```

## Architecture

```
CSV → profiler.py (structured profile: types, missing, outliers, correlations)
        ↓
      visualizer.py (auto-generated plotly charts based on column types)
        ↓
      groq_agent.py (LLM investigates via sandboxed run_pandas_query tool, then writes report)
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
  bypasses, then evaluates with a small built-in whitelist instead of the real
  `__builtins__`. See `sandbox.py`'s docstring for the specific bypass
  (`"{0.__class__}".format(df)`) that was found and patched during development —
  string-format-based dunder access doesn't show up as an `Attribute` node in the AST, so
  a naive "block dunder attribute access" check alone isn't sufficient. Code-injection is
  covered by the AST checks; resource exhaustion (a syntactically-safe expression that's
  computationally enormous — a huge array allocation, a runaway loop) is a separate attack
  class those checks don't touch, so each query runs in a forked child process with a
  wall-clock timeout (5s) and a memory cap (512MB), hard-killed if it exceeds either. Fork
  is used specifically because it gives the child a copy-on-write view of `df` instead of
  pickling it on every query — the trade-off is that forking from a multi-threaded process
  (which Streamlit's server is) carries a small, documented risk of a child deadlock; see
  the comment in `sandbox.py` for why that risk is acceptable here (the forked child does
  one fast thing and is hard-killed by the timeout regardless of whether it deadlocks).
- **PII detection is type-aware** to avoid false positives (e.g. a datetime column like
  `"2023-01-01"` structurally resembles a phone-number regex; the check skips
  pattern-matching for numeric/datetime columns and relies on column-name hints instead).
- **Deterministic data quality checks that don't rely on the LLM noticing.** Testing against
  a genuinely messy real-world commuter dataset (realistic entry errors: inconsistent
  station-name casing/whitespace, duplicated IDs, mixed numeric/text values, an invalid
  `"25:00"` timestamp) showed the agent catching some issues (a `999` sentinel value
  skewing an average) but missing others it should have caught reliably. The pattern:
  everything missed was something code can check for free, every time, at zero token
  cost — no reason to leave it to chance. Added four deterministic checks to `profiler.py`
  (`detect_category_normalization_issues`, `detect_duplicate_id_values`,
  `detect_mixed_numeric_text`, and per-value datetime-parse-failure tracking) so these are
  now caught in the profile itself, before the LLM is even involved. One design pitfall
  caught along the way: gating the duplicate-ID check behind the `id_like` type
  classification doesn't work, because a duplicate is exactly what drops a column's
  uniqueness ratio below the threshold that classifies it as `id_like` in the first place —
  fixed by computing uniqueness directly rather than relying on the type label.
- **Manual (not automatic) function calling.** Groq's SDK follows the same OpenAI-style
  tool-calling shape used by most providers, which supports auto-executing functions passed
  as tools — but we deliberately bypass that convenience and handle each tool call
  ourselves so every query goes through the sandbox's AST safety checks before it ever
  touches the real dataframe.
- **Bounded conversation context in the agentic loop.** Earlier versions rebuilt the full
  conversation history on every tool-call iteration — system prompt + full profile + every
  past tool call and its full raw result, growing without bound as the investigation ran
  longer. This was flagged in code review: by the 5th tool call, a single API request could
  carry 4,000-6,000+ input tokens just to decide whether a 6th call was needed, which
  disproportionately strains a free-tier token budget on later turns. Fixed by keeping only
  the most recent 2 tool-call turns in full and merging everything older into a single
  compact one-line-per-query summary — proven via tests that literally double the
  iteration count and assert the resulting message count is unchanged (`groq_agent.py`'s
  `_build_messages`). Also reduced the sandbox's default result verbosity
  (`MAX_RESULT_CHARS` 3000→1000, `max_rows` 30→12) since results were sized for human
  readability, not for repeatedly feeding back into an LLM.
- **Provider migration history.** The agent loop was originally prototyped against the
  Anthropic API, then Google Gemini's free tier, before settling on Groq — chosen for a
  more generous requests-per-minute limit (important since the agentic mode fires off
  several calls in a row) and simpler rate-limit error handling. `profiler.py`,
  `visualizer.py`, and `sandbox.py` never changed across any of these migrations; only the
  API client and message-formatting code did.
- **UI tested without a browser.** `test_streamlit_app.py` uses Streamlit's `AppTest`
  framework to drive real file uploads through the app headlessly, catching integration
  bugs (caching, render pipeline) that unit tests on the individual modules alone can't.

## Known limitations

- The sandbox's fork-based query isolation isn't available on Windows (`multiprocessing`
  has no `fork` start method there); queries fall back to running inline with no
  timeout/memory ceiling on that platform, though AST-level code-injection protection is
  unaffected. Fine for the primary deployment targets (Streamlit Community Cloud, HF
  Spaces, most CI), worth knowing if running locally on Windows.
- Single-user, local/dev-oriented app — no auth, no multi-tenant isolation. Not intended to
  be deployed as a public multi-user service without adding those.

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
- Fixed an unbounded conversation-context growth bug in the agentic loop (flagged in code
  review) by implementing a sliding context window that merges older tool-call turns into
  a single compact summary, capping per-request token growth at a constant regardless of
  investigation length — verified with tests that double the iteration count and assert
  message count is unchanged.
- Stress-tested the tool against a realistic messy dataset and found the LLM missed several
  data quality issues (inconsistent category casing/whitespace, a duplicated ID, an invalid
  timestamp) it should have caught; closed the gap by adding deterministic profiling checks
  that catch these classes of errors reliably and at zero token cost, rather than depending
  on model judgment for things code can verify directly.
- Achieved flat LLM token usage regardless of dataset size by sending only aggregated
  profile statistics, never raw rows, to the model.
- Shipped a Streamlit UI with headless integration tests (Streamlit `AppTest`) covering the
  full upload-to-report flow, reaching 67 passing tests across the project, enforced on
  every push via GitHub Actions CI.
