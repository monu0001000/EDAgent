"""
profiler.py
Phase 1: Pure-code data profiling — no AI involved.
Takes a pandas DataFrame and produces a structured "profile" dict:
column types, missing values, duplicates, outliers, correlations, and
basic stats. This structured profile is what we later feed to the LLM
(Phase 3) instead of the whole dataset — keeps token usage small and
avoids sending raw sensitive data unnecessarily.
"""

import pandas as pd
import numpy as np
import re


# ---------- Column type detection ----------

def detect_column_type(series: pd.Series) -> str:
    """Classify a column as numeric, datetime, categorical, text, or id_like."""
    s = series.dropna()
    if s.empty:
        return "empty"

    # Datetime check
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed = pd.to_datetime(s.sample(min(20, len(s)), random_state=0), errors="coerce")
            if parsed.notna().mean() > 0.8:
                return "datetime"
        except Exception:
            pass

    # Numeric check
    if pd.api.types.is_numeric_dtype(series):
        # ID-like: integer, unique, or sequential-looking
        if pd.api.types.is_integer_dtype(series) and s.nunique() / len(s) > 0.95:
            return "id_like"
        return "numeric"

    # Categorical vs free text vs id_like (object dtype)
    nunique_ratio = s.nunique() / len(s)
    avg_len = s.astype(str).str.len().mean()

    if nunique_ratio > 0.95 and avg_len < 30:
        return "id_like"
    if avg_len > 60:
        return "text"
    if s.nunique() <= max(20, len(s) * 0.05):
        return "categorical"
    return "text"


# ---------- PII detection (lightweight regex-based) ----------

PII_PATTERNS = {
    "email": re.compile(r"^[\w\.-]+@[\w\.-]+\.\w+$"),
    "phone": re.compile(r"^\+?\d[\d\-\s\(\)]{7,}\d$"),
    "ssn": re.compile(r"^\d{3}-\d{2}-\d{4}$"),
}


def detect_pii(series: pd.Series, col_name: str, col_type: str = "text") -> str | None:
    """Return a PII type label if column looks like it holds sensitive data.

    col_type is used to avoid false positives: e.g. a datetime column like
    "2023-01-01" structurally resembles a phone-number regex, so we skip
    pattern-based checks for datetime/numeric columns and rely on name hints only.
    """
    name_lower = col_name.lower()
    name_hints = {
        "email": ["email", "e-mail"],
        "phone": ["phone", "mobile", "contact_no"],
        "ssn": ["ssn", "social_security"],
        "name": ["first_name", "last_name", "full_name", "customer_name"],
        "address": ["address", "street"],
    }
    for pii_type, hints in name_hints.items():
        if any(h in name_lower for h in hints):
            return pii_type

    if col_type in ("datetime", "numeric"):
        return None  # avoid false positives from digit-pattern regexes on dates/numbers

    sample = series.dropna().astype(str).head(30)
    if sample.empty:
        return None
    for pii_type, pattern in PII_PATTERNS.items():
        match_ratio = sample.apply(lambda x: bool(pattern.match(x.strip()))).mean()
        if match_ratio > 0.7:
            return pii_type
    return None


# ---------- Outlier detection ----------

def detect_outliers_iqr(series: pd.Series) -> dict:
    """IQR-based outlier detection for a numeric column."""
    s = series.dropna()
    if len(s) < 5:
        return {"count": 0, "pct": 0.0, "lower_bound": None, "upper_bound": None}
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = s[(s < lower) | (s > upper)]
    return {
        "count": int(len(outliers)),
        "pct": round(len(outliers) / len(s) * 100, 2),
        "lower_bound": round(float(lower), 3),
        "upper_bound": round(float(upper), 3),
    }


# ---------- Main profiling function ----------

def profile_dataframe(df: pd.DataFrame) -> dict:
    """Produce a full structured profile of the dataframe."""
    profile = {
        "shape": {"rows": df.shape[0], "cols": df.shape[1]},
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": {},
    }

    for col in df.columns:
        series = df[col]
        col_type = detect_column_type(series)
        pii_type = detect_pii(series, col, col_type)
        missing_count = int(series.isna().sum())
        missing_pct = round(missing_count / len(series) * 100, 2) if len(series) else 0.0

        col_info = {
            "dtype": str(series.dtype),
            "inferred_type": col_type,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "n_unique": int(series.nunique(dropna=True)),
            "pii_flag": pii_type,
        }

        if col_type == "numeric":
            desc = series.describe()
            col_info["stats"] = {
                "mean": round(float(desc.get("mean", np.nan)), 3),
                "std": round(float(desc.get("std", np.nan)), 3),
                "min": round(float(desc.get("min", np.nan)), 3),
                "max": round(float(desc.get("max", np.nan)), 3),
                "median": round(float(series.median()), 3),
            }
            col_info["outliers"] = detect_outliers_iqr(series)

        elif col_type == "categorical":
            top_vals = series.value_counts(dropna=True).head(5)
            col_info["top_values"] = {str(k): int(v) for k, v in top_vals.items()}

        elif col_type == "datetime":
            parsed = pd.to_datetime(series, errors="coerce")
            col_info["date_range"] = {
                "min": str(parsed.min()),
                "max": str(parsed.max()),
            }

        profile["columns"][col] = col_info

    # Correlations among numeric columns
    numeric_cols = [c for c, info in profile["columns"].items() if info["inferred_type"] == "numeric"]
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr(numeric_only=True)
        strong_pairs = []
        for i, c1 in enumerate(numeric_cols):
            for c2 in numeric_cols[i + 1:]:
                val = corr_matrix.loc[c1, c2]
                if pd.notna(val) and abs(val) > 0.5:
                    strong_pairs.append({"col1": c1, "col2": c2, "correlation": round(float(val), 3)})
        profile["strong_correlations"] = sorted(strong_pairs, key=lambda x: -abs(x["correlation"]))
    else:
        profile["strong_correlations"] = []

    return profile


if __name__ == "__main__":
    # Quick smoke test with a synthetic messy dataset
    rng = np.random.default_rng(0)
    test_df = pd.DataFrame({
        "customer_id": range(1, 201),
        "email": [f"user{i}@test.com" for i in range(200)],
        "age": rng.integers(18, 90, 200).astype(float),
        "signup_date": pd.date_range("2023-01-01", periods=200, freq="D").astype(str),
        "plan": rng.choice(["basic", "pro", "enterprise"], 200),
        "monthly_spend": rng.normal(50, 15, 200),
    })
    test_df.loc[5:10, "age"] = np.nan
    test_df.loc[0, "monthly_spend"] = 5000  # outlier

    result = profile_dataframe(test_df)
    import json
    print(json.dumps(result, indent=2, default=str))
