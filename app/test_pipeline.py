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

from profiler import (
    profile_dataframe, detect_column_type, detect_pii, detect_outliers_iqr,
    detect_category_normalization_issues, detect_duplicate_id_values, detect_mixed_numeric_text,
)
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


# ---------- Category normalization issues ----------
# These mirror real issues found testing against a messy real-world CSV
# (train commuter data): "Bandra" vs "  Bandra  ", "High" vs "high" vs
# "MEDIUM" vs "Medium" — an LLM reasoning over aggregated stats missed all
# of these, but they're trivial for code to catch deterministically.

def test_category_normalization_detects_whitespace_variant():
    s = pd.Series(["Bandra", "Bandra", "  Bandra  ", "Andheri"])
    issues = detect_category_normalization_issues(s)
    assert len(issues) == 1
    assert issues[0]["normalized"] == "bandra"
    assert set(issues[0]["variants"]) == {"Bandra", "  Bandra  "}
    assert issues[0]["counts"]["Bandra"] == 2
    assert issues[0]["counts"]["  Bandra  "] == 1

def test_category_normalization_detects_case_variants():
    s = pd.Series(["High", "high", "Medium", "MEDIUM", "Medium", "Low"])
    issues = detect_category_normalization_issues(s)
    normalized_forms = {i["normalized"] for i in issues}
    assert "high" in normalized_forms
    assert "medium" in normalized_forms

def test_category_normalization_no_false_positive_on_clean_data():
    s = pd.Series(["High", "Medium", "Low", "High", "Medium"])
    assert detect_category_normalization_issues(s) == []

def test_category_normalization_does_not_catch_genuine_typos():
    # "Thane" vs "Thna" is a real typo, not a whitespace/case issue — this
    # detector deliberately does NOT try to catch it (that would need fuzzy
    # matching, which risks false positives on genuinely distinct short
    # names). Documenting this as a known, intentional limitation.
    s = pd.Series(["Thane", "Thna", "Thane"])
    assert detect_category_normalization_issues(s) == []


# ---------- Duplicate ID values ----------

def test_duplicate_id_values_detected():
    s = pd.Series(["TRN1000", "TRN1001", "TRN1014", "TRN1014", "TRN1016"])
    result = detect_duplicate_id_values(s)
    assert result["count"] == 1
    assert result["examples"][0]["value"] == "TRN1014"
    assert result["examples"][0]["occurrences"] == 2

def test_duplicate_id_values_none_when_all_unique():
    s = pd.Series(["A1", "A2", "A3"])
    result = detect_duplicate_id_values(s)
    assert result["count"] == 0
    assert result["examples"] == []


# ---------- Mixed numeric/text formatting ----------

def test_mixed_numeric_text_detected():
    s = pd.Series(["1", "2", "3", "Two", "4", "5"])
    result = detect_mixed_numeric_text(s)
    assert result is not None
    assert "Two" in result["non_numeric_examples"]
    assert 0.7 < result["numeric_ratio"] < 1.0  # 5/6 = 0.833

def test_mixed_numeric_text_not_flagged_when_fully_numeric():
    s = pd.Series(["1", "2", "3", "4"])
    assert detect_mixed_numeric_text(s) is None

def test_mixed_numeric_text_not_flagged_when_mostly_text():
    # If most values are genuinely non-numeric (below threshold), this is
    # just a normal text/categorical column, not a formatting inconsistency.
    s = pd.Series(["red", "blue", "green", "1"])
    assert detect_mixed_numeric_text(s, threshold=0.7) is None


# ---------- Integration: unparseable datetime values ----------

def test_profile_flags_unparseable_datetime_value_separately_from_missing():
    """A value that's present (not null) but fails to parse as a valid
    datetime — like '25:00', an invalid hour — must be surfaced distinctly
    from missing_count, which only reflects raw nulls. Without this, an
    unparseable-but-present value silently disappears from the profile.
    Uses enough rows that the single bad value doesn't itself push the
    column below detect_column_type's own datetime-detection threshold."""
    times = ["08:00", "09:15", "10:30", "11:45", "12:00",
             "13:15", "14:30", "15:45", "16:00", "17:15", "25:00"]
    df = pd.DataFrame({"scheduled_time": times})
    profile = profile_dataframe(df)
    col_info = profile["columns"]["scheduled_time"]
    assert col_info["inferred_type"] == "datetime"
    assert col_info["missing_count"] == 0  # nothing is null
    assert col_info["unparseable_values"]["count"] == 1
    assert "25:00" in col_info["unparseable_values"]["examples"]

def test_profile_integration_catches_all_real_world_messy_data_issues():
    """End-to-end check mirroring the actual messy dataset that surfaced
    these gaps: whitespace-padded categories, case-duplicate categories,
    a duplicated ID, and a mixed numeric/text column, all in one profile."""
    df = pd.DataFrame({
        "train_id": ["TRN1", "TRN2", "TRN3", "TRN3", "TRN4", "TRN5", "TRN6"],
        "station": ["Bandra", "  Bandra  ", "Andheri", "Andheri", "Dadar", "Thane", "Kalyan"],
        "crowd": ["High", "high", "Medium", "Medium", "Low", "Low", "High"],
        "platform": ["1", "2", "Two", "3", "4", "1", "2"],
    })
    profile = profile_dataframe(df)

    assert profile["columns"]["train_id"]["duplicate_id_values"]["count"] == 1
    assert len(profile["columns"]["station"]["category_normalization_issues"]) == 1
    assert len(profile["columns"]["crowd"]["category_normalization_issues"]) == 1
    assert profile["columns"]["platform"]["mixed_numeric_text"]["non_numeric_examples"] == ["Two"]

def test_duplicate_id_check_does_not_false_positive_on_continuous_numeric_columns():
    """Regression test: found via the eval harness (evaluate.py). A
    continuous float column can easily have a couple of coincidentally-
    equal rounded values with no data-entry error involved — that's not
    the same signal as a repeated identifier and shouldn't be flagged as
    one."""
    rng = np.random.default_rng(1)
    n = 150
    df = pd.DataFrame({
        "customer_id": range(1, n + 1),  # genuinely unique — must NOT be flagged
        "monthly_spend": rng.normal(50, 15, n).round(2),  # continuous — must NOT be flagged
    })
    profile = profile_dataframe(df)
    assert "duplicate_id_values" not in profile["columns"]["customer_id"]
    assert "duplicate_id_values" not in profile["columns"]["monthly_spend"]


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

# These two exercise the resource-exhaustion guard: expressions that pass
# every AST safety check (no imports, no dunder access, nothing on the
# blocklist) but are computationally hostile rather than malicious code.
# Skipped where fork() isn't available (e.g. Windows), since that's the
# only platform where the isolation itself is skipped too (see sandbox.py).
import sandbox as _sandbox_module

@pytest.mark.skipif(not _sandbox_module._FORK_AVAILABLE, reason="timeout/memory isolation requires fork()")
def test_sandbox_kills_queries_that_exceed_the_time_limit(small_df, monkeypatch):
    monkeypatch.setattr(_sandbox_module, "QUERY_TIMEOUT_SECONDS", 0.05)
    # Syntactically fine, no AST rule objects to it — just slow enough to
    # blow straight through a near-zero timeout.
    result = safe_query(small_df, "sum(range(10**8))")
    assert "time limit" in result.lower()

@pytest.mark.skipif(not _sandbox_module._FORK_AVAILABLE, reason="timeout/memory isolation requires fork()")
def test_sandbox_kills_queries_that_exceed_the_memory_limit(small_df, monkeypatch):
    monkeypatch.setattr(_sandbox_module, "QUERY_MEMORY_LIMIT_BYTES", 20 * 1024 * 1024)  # 20 MB cap
    result = safe_query(small_df, "np.zeros(10**8)")  # ~800 MB, well over the cap
    assert "memory" in result.lower() or "terminated" in result.lower()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
