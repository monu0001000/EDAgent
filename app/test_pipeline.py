"""
test_pipeline.py
Basic regression tests for the profiler + visualizer modules.
Run with: python3 -m pytest test_pipeline.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import pytest

from profiler import profile_dataframe, detect_column_type, detect_pii, detect_outliers_iqr
from visualizer import generate_charts
from sandbox import safe_query, UnsafeQueryError


# ---------- Fixtures ----------

@pytest.fixture
def messy_df():
    rng = np.random.default_rng(42)
    n = 150
    spend = rng.normal(50, 15, n)
    spend[0] = 5000  # injected outlier
    df = pd.DataFrame({
        "id": range(1, n + 1),
        "email": [f"user{i}@test.com" for i in range(n)],
        "age": rng.integers(18, 90, n).astype(float),
        "signup_date": pd.date_range("2023-01-01", periods=n, freq="D").astype(str),
        "plan": rng.choice(["basic", "pro", "enterprise"], n),
        "spend": spend,
        "tickets": spend / 10 + rng.normal(0, 1, n),  # correlated with spend
    })
    df.loc[3:8, "age"] = np.nan  # injected missing values
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # injected duplicate row
    return df


# ---------- Column type detection ----------

def test_numeric_column_detected():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0] * 10)
    assert detect_column_type(s) == "numeric"

def test_datetime_column_detected_from_strings():
    s = pd.Series(pd.date_range("2023-01-01", periods=50).astype(str))
    assert detect_column_type(s) == "datetime"

def test_categorical_column_detected():
    s = pd.Series(["a", "b", "c"] * 30)
    assert detect_column_type(s) == "categorical"

def test_id_like_column_detected():
    s = pd.Series(range(100))
    assert detect_column_type(s) == "id_like"

def test_empty_column_handled():
    s = pd.Series([None, None, None])
    assert detect_column_type(s) == "empty"


# ---------- PII detection ----------

def test_email_pii_detected_by_content():
    s = pd.Series([f"user{i}@test.com" for i in range(30)])
    assert detect_pii(s, "contact", col_type="id_like") == "email"

def test_pii_name_hint_overrides():
    s = pd.Series(["x"] * 30)
    assert detect_pii(s, "email_address", col_type="text") == "email"

def test_datetime_column_not_falsely_flagged_as_phone():
    # Regression test: dates like "2023-01-01" structurally resemble phone
    # number regexes (digits + dashes) — must not false-positive.
    s = pd.Series(pd.date_range("2023-01-01", periods=30).astype(str))
    assert detect_pii(s, "signup_date", col_type="datetime") is None


# ---------- Outlier detection ----------

def test_outliers_detected():
    s = pd.Series([10, 11, 12, 11, 10, 12, 500])  # 500 is a clear outlier
    result = detect_outliers_iqr(s)
    assert result["count"] >= 1

def test_no_outliers_in_uniform_data():
    s = pd.Series([10, 11, 12, 11, 10, 12, 11])
    result = detect_outliers_iqr(s)
    assert result["count"] == 0

def test_outliers_handle_tiny_series():
    s = pd.Series([1, 2])
    result = detect_outliers_iqr(s)
    assert result["count"] == 0  # too few points to compute IQR meaningfully


# ---------- Full profile integration ----------

def test_profile_dataframe_full_pipeline(messy_df):
    profile = profile_dataframe(messy_df)
    assert profile["shape"]["rows"] == len(messy_df)
    assert profile["duplicate_rows"] >= 1  # we injected one
    assert profile["columns"]["email"]["pii_flag"] == "email"
    assert profile["columns"]["age"]["missing_count"] >= 6
    assert profile["columns"]["signup_date"]["inferred_type"] == "datetime"
    assert profile["columns"]["signup_date"]["pii_flag"] is None  # regression check
    assert profile["columns"]["spend"]["outliers"]["count"] >= 1
    # spend and tickets should show up as strongly correlated
    corr_pairs = {(c["col1"], c["col2"]) for c in profile["strong_correlations"]}
    assert ("spend", "tickets") in corr_pairs

def test_profile_handles_empty_dataframe():
    df = pd.DataFrame()
    profile = profile_dataframe(df)
    assert profile["shape"]["rows"] == 0
    assert profile["columns"] == {}


# ---------- Visualizer ----------

def test_charts_generated_for_all_relevant_types(messy_df):
    profile = profile_dataframe(messy_df)
    charts = generate_charts(messy_df, profile)
    assert "dist_age" in charts
    assert "dist_spend" in charts
    assert "cat_plan" in charts
    assert "time_signup_date" in charts
    assert "correlation_heatmap" in charts
    # id_like columns (id, email) should NOT get charts
    assert not any(name.endswith("_id") for name in charts)
    assert not any("email" in name for name in charts)

def test_scatter_generated_for_strong_correlation(messy_df):
    profile = profile_dataframe(messy_df)
    charts = generate_charts(messy_df, profile)
    assert any("scatter_spend_vs_tickets" in name for name in charts)


# ---------- Sandbox safety ----------

@pytest.fixture
def small_df():
    return pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "x", "y", "z"]})

@pytest.mark.parametrize("query", [
    "df['a'].mean()",
    "df.groupby('b')['a'].sum()",
    "df[df['a'] > 2]",
    "df['a'].sum() + df['a'].mean()",
    "df.describe()",
    "df['b'].value_counts()",
    "df.corr(numeric_only=True)",
])
def test_sandbox_allows_legitimate_pandas_queries(small_df, query):
    result = safe_query(small_df, query)
    assert "not allowed" not in result.lower()

@pytest.mark.parametrize("query", [
    "__import__('os').system('echo hacked')",
    "open('/etc/passwd').read()",
    "df.__class__.__bases__",
    "exec('import os')",
    "eval('1+1')",
    "df.a = 999",
    "[x for x in ().__class__.__base__.__subclasses__()]",
    '"{0.__class__}".format(df)',
    '"{0.__class__.__bases__[0]}".format(df)',
    'f"{df.__class__}"',
    "getattr(df, '__class__')",
    "import os",
])
def test_sandbox_blocks_escape_attempts(small_df, query):
    with pytest.raises(UnsafeQueryError):
        safe_query(small_df, query)

def test_sandbox_truncates_large_results(small_df):
    # to_string(max_rows=30) already caps row count, so to actually exceed
    # MAX_RESULT_CHARS we need a result that's wide, not just tall.
    wide_df = pd.DataFrame({f"col_{i}": range(50) for i in range(200)})
    result = safe_query(wide_df, "df")
    assert "truncated" in result

def test_sandbox_handles_runtime_errors_gracefully(small_df):
    result = safe_query(small_df, "df['nonexistent_column']")
    assert "error" in result.lower()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
