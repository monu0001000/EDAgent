# EDAgent

An AI-powered data analysis agent: upload any CSV and it automatically profiles the data,
generates relevant visualizations, and uses an LLM (with tool-calling) to investigate
patterns and write a natural-language insights report — deciding what to dig into rather
than just summarizing fixed statistics. You can also ask it direct questions about specific
rows in your data ("is order #4471 likely to be returned?", "will customer C-8842 churn?")
and it reasons from comparable historical records rather than guessing. Domain-agnostic:
works the same way on transaction logs, customer data, sensor readings, or anything else —
it reads whatever columns are actually in your CSV rather than assuming what they mean.

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
| Ask-a-question mode | `app/groq_agent.py` (`answer_question`) | ✅ Done, tested — domain-agnostic Q&A grounded in historical rows |
| Streamlit UI | `app/streamlit_app.py` | ✅ Done, tested |
| Evaluation harness | `app/evaluate.py`, `app/eval_datasets.py` | ✅ Done, tested — scores reports against profiler ground truth |

**92/92 tests passing** across `test_pipeline.py`, `test_groq_agent.py`, `test_streamlit_app.py`, `test_evaluate.py`.

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
Defaults to `openai/gpt-oss-120b`, overridable via `GROQ_MODEL` in `.env` if that model
ever runs out of quota or is deprecated.

**Model migration note:** this project originally defaulted to `llama-3.3-70b-versatile`.
Groq deprecated it (along with `llama-3.1-8b-instant`) on August 16, 2026 and recommended
migrating to `openai/gpt-oss-120b` or `qwen/qwen3.6-27b`; this project moved to
`openai/gpt-oss-120b`. Nothing else needed to change — Groq's tool-calling response shape
(`tool_calls`, `message.content` for the final answer) is identical across their model
lineup, so `groq_agent.py`'s tool loop didn't need any changes, only the model string. If
you hit a `model_decommissioned` error on any Groq model in the future, check
https://console.groq.com/docs/deprecations for the current recommended replacement and
update `GROQ_MODEL` (or the default in `groq_agent.py`/`groq_insight_generator.py`)
accordingly — Groq's model lineup changes faster than most providers'.

## Run the app

```bash
cd app
streamlit run streamlit_app.py
```

Upload a CSV, review the auto-generated profile and charts, then either:
- **Generate a report** — pick **Agentic** (investigates via its own pandas queries first,
  slower and sharper) or **Single-shot** (one call, summarizes the pre-computed profile,
  faster and simpler).
- **Ask a specific question** — e.g. "is row X going to happen?", "which segment has the
  highest average Y?". The agent finds the relevant row(s) and comparable historical data,
  then answers grounded in what it actually finds — see "Ask-a-question mode" below for how
  it decides what counts as a fair comparison for whatever your columns actually are.


## Run individual pieces / tests

```bash
cd app
python3 profiler.py                 # profiler smoke test, no API key needed
python3 visualizer.py               # chart generation smoke test, no API key needed
python3 sandbox.py                  # sandbox security smoke test, no API key needed
python3 groq_insight_generator.py   # single-shot report — needs GROQ_API_KEY
python3 groq_agent.py               # agentic report — needs GROQ_API_KEY, watch it investigate
python3 evaluate.py                 # evaluation harness — needs GROQ_API_KEY, see Evaluation below

python3 -m pytest -v                # full test suite (92 tests)
```

## Architecture

```
CSV → profiler.py (structured profile: types, missing, outliers, correlations)
        ↓
      visualizer.py (auto-generated plotly charts based on column types)
        ↓
      groq_agent.py (LLM investigates via sandboxed run_pandas_query tool, then writes
                      a report OR answers one specific question — see below)
        ↓
      streamlit_app.py (upload → profile/charts → report or Q&A, in the browser)
```

Design choices worth noting:

- **One shared tool-calling loop, two tasks built on top of it.** `generate_insights_agentic`
  (writes a full report) and `answer_question` (answers one specific question) both call a
  single internal `_run_tool_loop` — same message-history bookkeeping, tool execution,
  retry/backoff, and forced-final-answer fallback either way. Only the system prompt and the
  initial user message differ per task. Originally `answer_question` was written by copying
  the report-generation loop and editing the prompt, which immediately produced two
  divergent copies of ~100 lines of message-bookkeeping logic; refactored into the shared
  loop before that drift could compound — the two tasks share a bug fix here for free
  instead of needing it applied twice.
- **Ask-a-question mode is deliberately domain-agnostic.** "Is train TRN1014 going to be
  late?", "will customer C-8842 churn?", and "is transaction TXN-991 fraudulent?" are the
  same underlying task: find the specific row, find comparable historical rows (same
  route/segment/category), look at the actual outcome distribution for those, and answer
  from that — not from guessing. `QA_SYSTEM_PROMPT` (`groq_agent.py`) never references a
  domain; it reads the profile's actual column names and the question text to work out what
  "comparable" means for whatever the uploaded CSV happens to contain. It's also explicitly
  instructed to state how much historical evidence its answer rests on and hedge accordingly
  — a pattern drawn from 2-3 rows is flagged as a hint, not asserted as a finding, and it
  cross-checks small groups against `category_normalization_issues` before trusting them
  (the same anomaly-detection instinct added to the report prompt after the 999/MEDIUM
  finding below — this is the same gap, just showing up in a different feature).
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

## Evaluation

`evaluate.py` runs either report generator across a set of synthetic
datasets in `eval_datasets.py` (six generated, plus a seventh loaded from
`data/local_train_commuter_data.csv` if present — the real messy dataset
that originally motivated `profiler.py`'s deterministic checks), each
built to contain a known data-quality issue (or, in one case, none at
all), and scores the resulting report against ground truth — not
hand-labeled, but computed by calling `profile_dataframe()` on the same
data, so the answer key can never drift out of sync with what the profiler
actually detects.

It's deliberately a narrow metric: **grounding, not creativity.** For every
issue the profiler already flagged deterministically (a duplicate ID, an
inconsistent category, a broken timestamp, a strong correlation), did the
final report actually mention it? A model that quietly drops a flagged
column from its report is failing the same way a model that never checked
would — the profiler's detection work is wasted if the report doesn't pass
it through. This intentionally does *not* try to score whether the model
found something insightful beyond the ground truth; that's a fuzzier
problem and a much worse fit for an automated score you can actually trust
run over run.

```bash
cd app
python3 evaluate.py                     # agentic mode, all cases
python3 evaluate.py --mode single_shot
python3 evaluate.py --case real_messy_commuter_data --quiet
```

Results are printed per-case (`column_recall`, `correlation_recall`, and
which specific columns got missed) and written to `reports/eval_results.json`
for a full record of what each run's reports actually said.

Building this surfaced a real bug before it was a problem in production:
the very first live run against the `outlier_and_correlation` and `clean`
cases showed two plain continuous numeric columns (`monthly_spend`,
`duration_sec`) getting flagged as `duplicate_id` — a handful of
coincidentally-equal rounded float values, not a broken identifier. Traced
it to `profile_dataframe`'s duplicate-value check running against every
column type rather than just identifier-shaped ones (`id_like`,
`categorical`, `text`); fixed by restricting the check to those types
(`profiler.py`), with a regression test (`test_duplicate_id_check_does_not_false_positive_on_continuous_numeric_columns`)
covering the exact case. Exactly the kind of thing the harness exists to
catch — a bug in the ground truth itself would have silently made every
future eval score look worse (or better) than the model's actual behavior
warranted.

A second, different-shaped finding came from running the actual app (not
just the harness) against the real messy commuter dataset. The agent's
report correctly listed every deterministic data-quality flag, but its
"Key Patterns" section presented a groupby result — `crowd_status ==
'MEDIUM'` has an average `delay_minutes` of 999 — as a normal finding,
when the profile it was already looking at showed `'MEDIUM'` occurs
exactly once in the data (a case-variant of the far more common
`'Medium'`, already flagged by `category_normalization_issues`) and
`delay_minutes.max` is also exactly 999 — i.e. that one row's value *is*
the group average, and 999 is a textbook sentinel/placeholder shape, not a
plausible delay. The model had every fact it needed already in the
profile and didn't connect them. Separately, the agent also spent its
final tool call on `df['delay_minutes'].corr(df['crowd_status'])`, which
fails outright (`.corr()` needs numeric columns on both sides), burning
its last chance to investigate further right as the investigation stopped.

Neither is a bug in the deterministic code — both are gaps in what the
prompt asked the model to do. Fixed in `groq_agent.py` and
`groq_insight_generator.py`:
- Raised `max_iterations` from 5 to 7 in the agentic loop, so one failed
  query doesn't consume the entire remaining investigation budget.
- Added explicit guidance to check a column's `inferred_type` before
  calling a numeric-only method like `.corr()` on it, and to recover with
  a different approach (e.g. `.groupby()`) rather than repeat the mistake.
- Added a required **"Anomalies Worth a Second Look"** report section,
  with explicit instruction to cross-check any dramatic aggregate against
  the group size and the already-detected quality flags before reporting
  it as a reliable pattern — the exact cross-reference the model skipped.
- Clarified that "concise" means no filler, not fewer findings — the
  original phrasing ("short bullets, not paragraphs") was apparently
  read as license to compress 5-6 distinct findings into 3-4 shallow ones.

Known limitation: this measures recall (did the report surface what's
true) but not precision (did it also invent things that aren't in the
profile). A model that pads every report with a paragraph of plausible-
sounding but unverified claims would still score well here. Adding a
hallucination/precision check — e.g. flagging specific numeric claims in
the report that don't trace back to any number in the profile or a tool
query result — is a natural next step.



## Next steps

- Precision/hallucination scoring in the eval harness (see the known limitation above)
- PDF/HTML export of the final report
- Deploy publicly (Streamlit Community Cloud or Hugging Face Spaces) for a live demo link

## Resume bullets

- Built EDAgent, an AI-powered data analysis tool that autonomously profiles, visualizes,
  and generates natural-language insight reports from arbitrary CSV files, using LLM
  tool-calling to dynamically drive exploratory analysis rather than a fixed pipeline.
- Added a domain-agnostic natural-language Q&A mode where the agent answers a question about
  a specific row (e.g. predicting a likely outcome) by finding comparable historical records
  and reasoning from their actual outcome distribution — with explicit confidence hedging
  based on how much historical evidence exists, rather than asserting a number from a
  handful of rows. Refactored the report-generation and Q&A code paths onto one shared
  tool-calling loop after noticing the second feature was about to duplicate ~100 lines of
  message-history/retry logic from the first.
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
- Built an evaluation harness that scores report quality against ground truth derived
  directly from the profiler's own deterministic checks (not hand-labeled), measuring
  whether the LLM's final report actually surfaces every issue the profiler already found —
  a grounding/faithfulness metric rather than a subjective quality score. Running it
  surfaced a real false-positive bug in the profiler itself (continuous numeric columns
  misflagged as broken identifiers), fixed with a regression test before it reached
  production.
- Achieved flat LLM token usage regardless of dataset size by sending only aggregated
  profile statistics, never raw rows, to the model.
- Shipped a Streamlit UI with headless integration tests (Streamlit `AppTest`) covering the
  full upload-to-report flow, reaching 92 passing tests across the project, enforced on
  every push via GitHub Actions CI.
