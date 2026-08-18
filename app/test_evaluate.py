"""
test_evaluate.py
Tests for evaluate.py's ground-truth extraction and scoring logic. These
are pure functions over profile dicts / report strings, so none of this
needs a live Groq call — same philosophy as test_groq_agent.py's mocked
agent loop tests, just applied to the eval harness itself.
"""

import pandas as pd
import pytest

from evaluate import extract_ground_truth, score_report
from profiler import profile_dataframe
from eval_datasets import get_eval_cases


def test_extract_ground_truth_finds_category_normalization_flag():
    df = pd.DataFrame({
        "city": ["Mumbai", "mumbai", "  Mumbai  ", "Delhi", "Delhi", "Delhi", "Delhi"] * 5,
        "id": range(35),
    })
    profile = profile_dataframe(df)
    gt = extract_ground_truth(profile)
    assert "category_normalization" in gt["flagged_columns"].get("city", [])

def test_extract_ground_truth_finds_duplicate_id_flag():
    order_id = list(range(1000, 1050))
    order_id[5] = order_id[6]  # force a duplicate
    df = pd.DataFrame({"order_id": order_id, "item": ["widget"] * 50})
    profile = profile_dataframe(df)
    gt = extract_ground_truth(profile)
    assert "duplicate_id" in gt["flagged_columns"].get("order_id", [])

def test_extract_ground_truth_finds_correlated_pairs():
    df = pd.DataFrame({
        # float dtype so these classify as "numeric", not "id_like"
        # (id_like requires an integer dtype — see profiler.detect_column_type)
        "x": [float(i) for i in range(50)],
        "y": [float(i * 2) for i in range(50)],  # perfectly correlated
    })
    profile = profile_dataframe(df)
    gt = extract_ground_truth(profile)
    pair_cols = {(p["col1"], p["col2"]) for p in gt["correlated_pairs"]}
    assert ("x", "y") in pair_cols

def test_extract_ground_truth_empty_when_nothing_flagged():
    df = pd.DataFrame({
        "id": range(1, 51),
        "plan": ["basic", "pro"] * 25,
    })
    profile = profile_dataframe(df)
    gt = extract_ground_truth(profile)
    assert gt["flagged_columns"] == {}
    assert gt["correlated_pairs"] == []

def test_score_report_full_recall_when_all_flagged_columns_mentioned():
    ground_truth = {
        "flagged_columns": {"city": ["category_normalization"], "order_id": ["duplicate_id"]},
        "correlated_pairs": [],
    }
    report = "## Data Quality Issues\n- The `city` column has inconsistent casing.\n- `order_id` has a duplicate value."
    scores = score_report(report, ground_truth)
    assert scores["column_recall"] == 1.0
    assert scores["matched_columns"] == {"city": True, "order_id": True}

def test_score_report_partial_recall_when_one_column_missed():
    ground_truth = {
        "flagged_columns": {"city": ["category_normalization"], "order_id": ["duplicate_id"]},
        "correlated_pairs": [],
    }
    report = "## Data Quality Issues\n- The `city` column has inconsistent casing."
    scores = score_report(report, ground_truth)
    assert scores["column_recall"] == 0.5
    assert scores["matched_columns"] == {"city": True, "order_id": False}

def test_score_report_correlation_recall_requires_both_columns_mentioned():
    ground_truth = {
        "flagged_columns": {},
        "correlated_pairs": [{"col1": "tenure_months", "col2": "monthly_spend", "correlation": 0.9}],
    }
    only_one_mentioned = "Monthly spend varies a lot across customers."
    assert score_report(only_one_mentioned, ground_truth)["correlation_recall"] == 0.0

    both_mentioned = "tenure_months is strongly correlated with monthly_spend."
    assert score_report(both_mentioned, ground_truth)["correlation_recall"] == 1.0

def test_score_report_vacuous_perfect_score_when_nothing_to_find():
    ground_truth = {"flagged_columns": {}, "correlated_pairs": []}
    scores = score_report("Nothing notable found in this dataset.", ground_truth)
    assert scores["column_recall"] == 1.0
    assert scores["correlation_recall"] == 1.0

def test_score_report_handles_none_report_text_without_crashing():
    ground_truth = {"flagged_columns": {"city": ["category_normalization"]}, "correlated_pairs": []}
    scores = score_report(None, ground_truth)
    assert scores["column_recall"] == 0.0

@pytest.mark.parametrize("case", get_eval_cases(), ids=lambda c: c["name"])
def test_eval_cases_produce_a_valid_ground_truth_shape(case):
    """Sanity check on the eval dataset fixtures themselves: every case
    must produce a well-formed ground truth dict from the real profiler,
    whatever it happens to flag (this deliberately doesn't assert which
    specific columns get flagged — that's exactly what would break the
    moment profiler.py's detection logic legitimately changes)."""
    profile = profile_dataframe(case["df"])
    gt = extract_ground_truth(profile)
    assert isinstance(gt["flagged_columns"], dict)
    assert isinstance(gt["correlated_pairs"], list)

def test_real_messy_commuter_data_case_is_loaded():
    """The real_messy_commuter_data case is optional (skipped if the CSV
    isn't present — see eval_datasets.py), but if it IS present, it should
    actually flag every issue class it's meant to demonstrate."""
    cases = {c["name"]: c for c in get_eval_cases()}
    if "real_messy_commuter_data" not in cases:
        pytest.skip("data/local_train_commuter_data.csv not present")

    profile = profile_dataframe(cases["real_messy_commuter_data"]["df"])
    gt = extract_ground_truth(profile)
    assert "category_normalization" in gt["flagged_columns"].get("station_from", [])
    assert "category_normalization" in gt["flagged_columns"].get("crowd_status", [])
    assert "duplicate_id" in gt["flagged_columns"].get("train_id", [])
    assert "mixed_numeric_text" in gt["flagged_columns"].get("platform", [])
    assert "unparseable_datetime" in gt["flagged_columns"].get("scheduled_departure", [])
