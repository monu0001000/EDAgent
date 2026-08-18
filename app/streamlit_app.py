"""
streamlit_app.py
EDAgent — the UI layer tying together profiler.py, visualizer.py, and
groq_agent.py / groq_insight_generator.py into a single app: upload a CSV,
see the auto-generated profile and charts, then run either the single-shot
or agentic report generator and watch what it finds.

Run with:
    cd app
    streamlit run streamlit_app.py
"""

import os
import io
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from profiler import profile_dataframe
from visualizer import generate_charts
from sandbox import UnsafeQueryError

load_dotenv()

st.set_page_config(page_title="EDAgent", page_icon="🔎", layout="wide")


# ---------------------------------------------------------------------------
# Styling — dark, focused palette. Signature element is the "investigation
# log": a terminal-styled panel that shows the agent's actual pandas queries
# as it runs them, since the whole point of this project is that the agent
# decides what to investigate rather than just summarizing fixed stats.
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --bg: #10131A;
        --panel: #1A1F2B;
        --panel-border: #2A3040;
        --text: #E7E9EE;
        --text-dim: #8A93A6;
        --accent: #F2B84B;
        --accent-dim: #7A6633;
        --data-blue: #6FB7E0;
    }

    .stApp { background-color: var(--bg); color: var(--text); }

    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }
    p, div, span, label { font-family: 'Inter', sans-serif; }

    .eda-hero {
        display: flex; align-items: baseline; gap: 0.6rem;
        border-bottom: 1px solid var(--panel-border);
        padding-bottom: 1.1rem; margin-bottom: 1.6rem;
    }
    .eda-hero h1 { font-size: 2.1rem; margin: 0; color: var(--text); }
    .eda-hero .eda-tag { color: var(--text-dim); font-size: 0.95rem; }

    .eda-stat-card {
        background: var(--panel); border: 1px solid var(--panel-border);
        border-radius: 6px; padding: 0.9rem 1.1rem;
    }
    .eda-stat-card .eda-stat-value {
        font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; color: var(--accent);
    }
    .eda-stat-card .eda-stat-label {
        color: var(--text-dim); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.04em;
    }

    /* The signature element: terminal-style investigation log */
    .eda-log {
        background: #0B0E13; border: 1px solid var(--panel-border); border-radius: 6px;
        padding: 0.9rem 1rem; font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem;
        max-height: 340px; overflow-y: auto;
    }
    .eda-log-line { margin-bottom: 0.55rem; }
    .eda-log-query { color: var(--accent); }
    .eda-log-result { color: var(--text-dim); white-space: pre-wrap; margin-top: 0.15rem; }
    .eda-log-prefix { color: var(--data-blue); }

    .eda-pii-badge {
        display: inline-block; background: #3A2418; color: var(--accent);
        border: 1px solid var(--accent-dim); border-radius: 4px;
        font-size: 0.7rem; padding: 0.05rem 0.4rem; margin-left: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def _run_profile(file_bytes: bytes) -> dict:
    df = _load_csv(file_bytes)
    return profile_dataframe(df)


def _stat_card(label: str, value) -> str:
    return f"""<div class="eda-stat-card">
        <div class="eda-stat-value">{value}</div>
        <div class="eda-stat-label">{label}</div>
    </div>"""


def _render_log(tool_calls: list[dict]):
    if not tool_calls:
        st.markdown(
            '<div class="eda-log"><span class="eda-log-result">No queries run yet.</span></div>',
            unsafe_allow_html=True,
        )
        return
    lines = []
    for i, tc in enumerate(tool_calls, 1):
        result_preview = tc["result"][:300]
        lines.append(
            f'<div class="eda-log-line">'
            f'<span class="eda-log-prefix">[{i}] &gt;&gt;&gt;</span> '
            f'<span class="eda-log-query">{tc["expression"]}</span>'
            f'<div class="eda-log-result">{result_preview}</div>'
            f'</div>'
        )
    st.markdown(f'<div class="eda-log">{"".join(lines)}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="eda-hero">
        <h1>🔎 EDAgent</h1>
        <span class="eda-tag">upload a CSV — it profiles, visualizes, and investigates on its own</span>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded = st.file_uploader("Upload a CSV", type=["csv"])

if uploaded is None:
    st.info("Drop a CSV in to get started. Nothing is sent anywhere until you upload.")
    st.stop()

file_bytes = uploaded.getvalue()

try:
    df = _load_csv(file_bytes)
except Exception as e:
    st.error(f"Couldn't read that file as a CSV: {e}")
    st.stop()

with st.spinner("Profiling dataset..."):
    profile = _run_profile(file_bytes)

# ---------------------------------------------------------------------------
# Profile summary
# ---------------------------------------------------------------------------

st.markdown("### Overview")
n_missing_cols = sum(1 for c in profile["columns"].values() if c["missing_count"] > 0)
n_pii_cols = sum(1 for c in profile["columns"].values() if c["pii_flag"])

cols = st.columns(5)
with cols[0]:
    st.markdown(_stat_card("Rows", profile["shape"]["rows"]), unsafe_allow_html=True)
with cols[1]:
    st.markdown(_stat_card("Columns", profile["shape"]["cols"]), unsafe_allow_html=True)
with cols[2]:
    st.markdown(_stat_card("Duplicate rows", profile["duplicate_rows"]), unsafe_allow_html=True)
with cols[3]:
    st.markdown(_stat_card("Columns w/ missing data", n_missing_cols), unsafe_allow_html=True)
with cols[4]:
    st.markdown(_stat_card("PII columns flagged", n_pii_cols), unsafe_allow_html=True)

with st.expander("Column details"):
    for col, info in profile["columns"].items():
        pii_badge = f'<span class="eda-pii-badge">PII: {info["pii_flag"]}</span>' if info["pii_flag"] else ""
        st.markdown(
            f'**`{col}`** — {info["inferred_type"]}, '
            f'{info["missing_count"]} missing ({info["missing_pct"]}%), '
            f'{info["n_unique"]} unique {pii_badge}',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

st.markdown("### Charts")
with st.spinner("Generating charts..."):
    charts = generate_charts(df, profile)

if not charts:
    st.caption("No chartable columns detected.")
else:
    chart_names = list(charts.keys())
    n_cols = 2
    for i in range(0, len(chart_names), n_cols):
        row_names = chart_names[i:i + n_cols]
        row_cols = st.columns(len(row_names))
        for col_widget, name in zip(row_cols, row_names):
            with col_widget:
                st.plotly_chart(charts[name], width="stretch", key=f"chart_{name}")

# ---------------------------------------------------------------------------
# AI report
# ---------------------------------------------------------------------------

st.markdown("### AI Report")

if not os.environ.get("GROQ_API_KEY"):
    st.warning(
        "No `GROQ_API_KEY` found. Add a free key from https://console.groq.com/keys "
        "to your `.env` file to generate a report.",
        icon="🔑",
    )

mode = st.radio(
    "Report mode",
    ["Agentic (investigates, then reports)", "Single-shot (summarizes profile only)"],
    horizontal=True,
)

if st.button("Generate report", type="primary", disabled=not os.environ.get("GROQ_API_KEY")):
    if "Agentic" in mode:
        from groq_agent import generate_insights_agentic

        with st.spinner("Agent is investigating the dataset..."):
            try:
                result = generate_insights_agentic(df, profile, verbose=False)
            except UnsafeQueryError as e:
                st.error(f"Sandbox rejected a query unexpectedly: {e}")
                st.stop()
            except Exception as e:
                st.error(f"Report generation failed: {e}")
                st.stop()

        st.markdown("**Investigation log**")
        _render_log(result["tool_calls"])
        st.markdown("**Report**")
        st.markdown(result["report"])
    else:
        from groq_insight_generator import generate_insights

        with st.spinner("Generating summary..."):
            try:
                report = generate_insights(profile, list(charts.keys()))
            except Exception as e:
                st.error(f"Report generation failed: {e}")
                st.stop()
        st.markdown(report)

# ---------------------------------------------------------------------------
# Ask a question — same agentic tool loop as the report, but aimed at one
# specific question instead of a general investigation. Deliberately
# domain-agnostic: works the same way whether the uploaded CSV is about
# trains, customers, transactions, or anything else — see
# groq_agent.QA_SYSTEM_PROMPT for how it decides what "comparable
# historical rows" means for whatever columns are actually present.
# ---------------------------------------------------------------------------

st.markdown("### Ask a Question")
st.caption(
    "Ask about a specific row or a general pattern — e.g. \"is train TRN1014 going to be "
    "late?\", \"will customer C-8842 churn?\", \"which category has the highest average spend?\". "
    "The agent looks at comparable historical rows in your data and answers from that, stating "
    "how much evidence it's actually based on."
)

if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

question = st.text_input(
    "Your question",
    placeholder="e.g. is train TRN1014 going to be late?",
    label_visibility="collapsed",
    disabled=not os.environ.get("GROQ_API_KEY"),
)

if st.button("Ask", disabled=not os.environ.get("GROQ_API_KEY") or not question.strip()):
    from groq_agent import answer_question

    with st.spinner("Looking through the data..."):
        try:
            result = answer_question(df, profile, question, verbose=False)
        except UnsafeQueryError as e:
            st.error(f"Sandbox rejected a query unexpectedly: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Couldn't answer that: {e}")
            st.stop()

    st.session_state.qa_history.insert(0, {
        "question": question,
        "answer": result["answer"],
        "tool_calls": result["tool_calls"],
    })

for qa in st.session_state.qa_history:
    with st.container(border=True):
        st.markdown(f"**Q: {qa['question']}**")
        st.markdown(qa["answer"])
        if qa["tool_calls"]:
            with st.expander(f"Investigation log ({len(qa['tool_calls'])} quer{'y' if len(qa['tool_calls']) == 1 else 'ies'})"):
                _render_log(qa["tool_calls"])