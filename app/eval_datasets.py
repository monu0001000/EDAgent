"""
eval_datasets.py
Synthetic test datasets for the evaluation harness (evaluate.py). Each case
injects one specific class of real-world data-quality issue — the same
classes profiler.py's deterministic checks exist to catch (see README's
"Deterministic data quality checks" section for the messy real-world dataset
that motivated them) — plus one fully clean dataset to check the agent
doesn't hallucinate problems that aren't there.

Deliberately does NOT hand-write the expected findings here. Ground truth
for each case is derived by evaluate.py running profile_dataframe() on
these dataframes directly, so the "answer key" can never drift out of sync
with what profiler.py actually detects — there's exactly one source of
truth for what counts as a flagged issue, and it's the same code path the
real app uses.

All datasets are built from a fixed seed, so re-running the harness always
scores against the same data.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
DATA_DIR = Path(__file__).parent.parent / "data"


def _messy_categories_df() -> pd.DataFrame:
    """A categorical column with whitespace/case variants of the same
    real-world value — the "Bandra" vs "  Bandra  " class of issue."""
    rng = np.random.default_rng(SEED)
    n = 150
    variants = ["Mumbai", "mumbai", "  Mumbai  ", "Delhi", "DELHI", "delhi ", "Chennai"]
    weights = [0.30, 0.10, 0.10, 0.20, 0.10, 0.05, 0.15]
    city = rng.choice(variants, n, p=weights)
    return pd.DataFrame({
        "record_id": range(1, n + 1),
        "city": city,
        "order_value": rng.normal(500, 120, n).round(2),
    })


def _duplicate_ids_df() -> pd.DataFrame:
    """An id-like column that should be unique per row but has a handful
    of accidental repeats — a copy-paste-style data entry error."""
    rng = np.random.default_rng(SEED)
    n = 150
    order_id = list(range(1000, 1000 + n))
    # Force a few duplicates in what otherwise looks like a unique key.
    order_id[10] = order_id[50]
    order_id[75] = order_id[76]
    return pd.DataFrame({
        "order_id": order_id,
        "item": rng.choice(["widget", "gadget", "gizmo"], n),
        "quantity": rng.integers(1, 10, n),
    })


def _mixed_numeric_text_df() -> pd.DataFrame:
    """A column that's almost entirely numeric-as-string but has a
    minority of values spelled out or formatted differently."""
    rng = np.random.default_rng(SEED)
    n = 150
    platform = rng.choice(["1", "2", "3"], n).astype(object)
    # ~10% of values get replaced with a non-numeric spelling of the same thing.
    spelled = {"1": "One", "2": "Two", "3": "Three"}
    bad_idx = rng.choice(n, size=max(1, n // 10), replace=False)
    for i in bad_idx:
        platform[i] = spelled[platform[i]]
    return pd.DataFrame({
        "session_id": range(1, n + 1),
        "platform": platform,
        "duration_sec": rng.normal(300, 60, n).round(1),
    })


def _bad_datetime_df() -> pd.DataFrame:
    """A datetime column with a mix of valid dates/timestamps and a few
    genuinely invalid ones (e.g. an impossible "25:00" hour) that would
    silently vanish from a naive profile instead of being flagged."""
    rng = np.random.default_rng(SEED)
    n = 150
    timestamps = pd.date_range("2024-01-01", periods=n, freq="h").astype(str).tolist()
    timestamps[3] = "2024-01-01 25:00:00"   # invalid hour
    timestamps[40] = "2024-13-45 10:00:00"  # invalid month/day
    return pd.DataFrame({
        "log_id": range(1, n + 1),
        "event_time": timestamps,
        "status_code": rng.choice([200, 200, 200, 404, 500], n),
    })


def _outlier_and_correlation_df() -> pd.DataFrame:
    """Two numeric columns with a genuine strong correlation, plus a
    deliberate outlier in one of them. The outlier is sized to trip the
    IQR check without being so extreme it wrecks the Pearson correlation
    itself (an early version used 9999, which dragged the correlation
    down to ~0.03 — a single wild point is enough to erase the very signal
    this case is meant to test for)."""
    rng = np.random.default_rng(SEED)
    n = 150
    tenure_months = rng.integers(1, 60, n).astype(float)
    monthly_spend = tenure_months * 8 + rng.normal(0, 15, n)
    monthly_spend[0] = 700  # clearly above the IQR upper bound, correlation stays intact
    return pd.DataFrame({
        "customer_id": range(1, n + 1),
        "tenure_months": tenure_months,
        "monthly_spend": monthly_spend.round(2),
    })


def _clean_df() -> pd.DataFrame:
    """No injected issues at all: consistent categories, unique IDs,
    plausible numeric ranges, valid dates. Tests whether the agent stays
    grounded and reports "nothing notable" instead of inventing problems
    to sound thorough."""
    rng = np.random.default_rng(SEED)
    n = 150
    return pd.DataFrame({
        "user_id": range(1, n + 1),
        "plan": rng.choice(["basic", "pro", "enterprise"], n, p=[0.6, 0.3, 0.1]),
        "signup_date": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
        "monthly_spend": rng.normal(45, 10, n).round(2).clip(min=5),
    })


def _real_messy_commuter_df() -> pd.DataFrame | None:
    """The actual real-world messy dataset referenced throughout the
    README (station-name whitespace/case variants, a duplicated train_id,
    an invalid "25:00" departure time, a mixed numeric/text platform
    column, and a 999-minute sentinel delay value hiding inside a
    single-row 'MEDIUM' crowd_status category). Loaded from disk rather
    than generated, so it's optional: returns None if the file isn't
    present rather than failing the whole eval run over one missing case."""
    path = DATA_DIR / "local_train_commuter_data.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def get_eval_cases() -> list[dict]:
    """Returns the fixed set of {"name", "description", "df"} evaluation
    cases. `evaluate.py` computes ground truth for each by profiling `df`
    directly — this module only owns the data, not the answer key."""
    cases = [
        {
            "name": "messy_categories",
            "description": "A 'city' column with whitespace/case variants of the same value.",
            "df": _messy_categories_df(),
        },
        {
            "name": "duplicate_ids",
            "description": "An 'order_id' column that should be unique but has repeated values.",
            "df": _duplicate_ids_df(),
        },
        {
            "name": "mixed_numeric_text",
            "description": "A 'platform' column that's mostly numeric-as-string with some spelled-out exceptions.",
            "df": _mixed_numeric_text_df(),
        },
        {
            "name": "bad_datetime",
            "description": "An 'event_time' column with a couple of structurally invalid timestamps.",
            "df": _bad_datetime_df(),
        },
        {
            "name": "outlier_and_correlation",
            "description": "A genuine strong correlation plus an IQR-flagged outlier in 'monthly_spend'.",
            "df": _outlier_and_correlation_df(),
        },
        {
            "name": "clean",
            "description": "No injected issues — checks the agent doesn't hallucinate problems.",
            "df": _clean_df(),
        },
    ]

    real_df = _real_messy_commuter_df()
    if real_df is not None:
        cases.append({
            "name": "real_messy_commuter_data",
            "description": (
                "The real-world messy dataset that originally motivated profiler.py's "
                "deterministic checks — every issue class in one 25-row file, including a "
                "999-minute sentinel value hiding inside a single-row 'MEDIUM' category."
            ),
            "df": real_df,
        })

    return cases


if __name__ == "__main__":
    for case in get_eval_cases():
        print(f"{case['name']}: {case['df'].shape[0]} rows x {case['df'].shape[1]} cols — {case['description']}")
