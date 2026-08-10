"""
visualizer.py
Phase 2: Auto-visualization — takes a DataFrame + the profile dict from
profiler.py and generates a set of relevant plotly figures automatically,
based on each column's inferred type. No AI involved yet; this is
deterministic "if numeric -> histogram" style logic. The LLM (Phase 3)
will later pick which of these charts are worth highlighting in the report.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _numeric_chart(df: pd.DataFrame, col: str) -> go.Figure:
    fig = px.histogram(df, x=col, marginal="box", title=f"Distribution of {col}")
    fig.update_layout(bargap=0.05)
    return fig


def _categorical_chart(df: pd.DataFrame, col: str, top_n: int = 10) -> go.Figure:
    counts = df[col].value_counts(dropna=True).head(top_n).reset_index()
    counts.columns = [col, "count"]
    fig = px.bar(counts, x=col, y="count", title=f"Top categories in {col}")
    return fig


def _datetime_chart(df: pd.DataFrame, col: str) -> go.Figure | None:
    parsed = pd.to_datetime(df[col], errors="coerce")
    if parsed.notna().sum() < 2:
        return None
    counts = parsed.dt.to_period("M").value_counts().sort_index()
    counts.index = counts.index.astype(str)
    fig = px.line(x=counts.index, y=counts.values,
                   title=f"Record count over time ({col})",
                   labels={"x": col, "y": "count"})
    return fig


def _correlation_heatmap(df: pd.DataFrame, numeric_cols: list[str]) -> go.Figure | None:
    if len(numeric_cols) < 2:
        return None
    corr = df[numeric_cols].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", title="Correlation heatmap", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    return fig


def _scatter_for_pair(df: pd.DataFrame, col1: str, col2: str) -> go.Figure:
    fig = px.scatter(df, x=col1, y=col2, title=f"{col1} vs {col2}", trendline="ols" if len(df) < 5000 else None)
    return fig


def generate_charts(df: pd.DataFrame, profile: dict, max_scatter_pairs: int = 3) -> dict:
    """
    Returns a dict of {chart_name: plotly.graph_objects.Figure}.
    Only generates charts for columns where it makes sense (skips id_like,
    text, and empty columns; skips categorical columns with too many
    unique values to avoid unreadable bar charts).
    """
    charts = {}
    numeric_cols = []

    for col, info in profile["columns"].items():
        col_type = info["inferred_type"]

        if col_type == "numeric":
            numeric_cols.append(col)
            charts[f"dist_{col}"] = _numeric_chart(df, col)

        elif col_type == "categorical":
            charts[f"cat_{col}"] = _categorical_chart(df, col)

        elif col_type == "datetime":
            fig = _datetime_chart(df, col)
            if fig is not None:
                charts[f"time_{col}"] = fig

        # id_like and text columns intentionally skipped — not useful to visualize directly

    # Correlation heatmap across all numeric columns
    heatmap = _correlation_heatmap(df, numeric_cols)
    if heatmap is not None:
        charts["correlation_heatmap"] = heatmap

    # Scatter plots for the strongest correlated pairs (from profile, capped)
    for pair in profile.get("strong_correlations", [])[:max_scatter_pairs]:
        c1, c2 = pair["col1"], pair["col2"]
        charts[f"scatter_{c1}_vs_{c2}"] = _scatter_for_pair(df, c1, c2)

    return charts


if __name__ == "__main__":
    import numpy as np
    from profiler import profile_dataframe

    rng = np.random.default_rng(0)
    n = 200
    monthly_spend = rng.normal(50, 15, n)
    test_df = pd.DataFrame({
        "customer_id": range(1, n + 1),
        "age": rng.integers(18, 90, n).astype(float),
        "signup_date": pd.date_range("2023-01-01", periods=n, freq="D").astype(str),
        "plan": rng.choice(["basic", "pro", "enterprise"], n),
        "monthly_spend": monthly_spend,
        "support_tickets": (monthly_spend / 10 + rng.normal(0, 1, n)),  # correlated with spend
    })

    profile = profile_dataframe(test_df)
    charts = generate_charts(test_df, profile)
    print(f"Generated {len(charts)} charts:")
    for name in charts:
        print(f"  - {name}")
