"""
evaluate.py
Evaluation harness for the report generators (groq_agent.py's agentic mode
and groq_insight_generator.py's single-shot mode).

This deliberately does NOT try to build a general "is this insight good"
judge — that's a much harder, fuzzier problem, and a bad fit for a metric
you can trust. Instead it answers a narrower, directly-checkable question:

    For every data-quality issue the profiler already found
    deterministically, does the model's final report actually surface it?

A model that silently drops a flagged duplicate-ID column from its report
is failing just as badly as one that never checked in the first place —
profiler.py doing the detection work is wasted if the report doesn't pass
it through. This is a grounding/faithfulness metric, not a creativity
metric: it measures whether the model reports what's already known to be
true, not whether it found anything beyond that on its own.

Ground truth for each of eval_datasets.py's cases is derived by running
profile_dataframe() directly — there's exactly one source of truth for
what counts as a flagged issue, and it's the same code path the real app
uses, so the answer key can't drift out of sync with the profiler.

Run with (needs GROQ_API_KEY):
    python3 evaluate.py                    # agentic mode, all cases
    python3 evaluate.py --mode single_shot
    python3 evaluate.py --case duplicate_ids
"""

import argparse
import json
import os
import sys
from pathlib import Path

from eval_datasets import get_eval_cases
from profiler import profile_dataframe

REPORTS_DIR = Path(__file__).parent.parent / "reports"


# ---------------------------------------------------------------------------
# Ground truth extraction — the only source of truth is profile_dataframe()
# ---------------------------------------------------------------------------

def extract_ground_truth(profile: dict) -> dict:
    """Pull every deterministically-flagged issue out of a profile dict.
    Returns {"flagged_columns": {col: [flag_type, ...]}, "correlated_pairs":
    [{"col1", "col2", "correlation"}]}. Column-level flag types:
    category_normalization, duplicate_id, mixed_numeric_text,
    unparseable_datetime, pii, outlier."""
    flagged_columns: dict[str, list[str]] = {}
    for col, info in profile.get("columns", {}).items():
        flags = []
        if info.get("category_normalization_issues"):
            flags.append("category_normalization")
        if info.get("duplicate_id_values"):
            flags.append("duplicate_id")
        if info.get("mixed_numeric_text"):
            flags.append("mixed_numeric_text")
        if info.get("unparseable_values"):
            flags.append("unparseable_datetime")
        if info.get("pii_flag"):
            flags.append("pii")
        if info.get("outliers", {}).get("count", 0) > 0:
            flags.append("outlier")
        if flags:
            flagged_columns[col] = flags

    correlated_pairs = list(profile.get("strong_correlations", []))
    return {"flagged_columns": flagged_columns, "correlated_pairs": correlated_pairs}


# ---------------------------------------------------------------------------
# Scoring — does the report text actually mention what ground truth flagged?
# ---------------------------------------------------------------------------

def score_report(report_text: str, ground_truth: dict) -> dict:
    """Grounding score: for every flagged column, is that column name
    mentioned anywhere in the report? For every strong correlation, are
    BOTH column names mentioned? Simple substring matching, not semantic
    matching — deliberately conservative, so a passing score is a real
    signal, not just the judge being generous. Known limitation: this
    checks recall (did it surface what's true) not precision (did it also
    invent things that aren't) — see README's eval harness section."""
    report_lower = (report_text or "").lower()

    flagged_columns = ground_truth["flagged_columns"]
    matched_columns = {
        col: (col.lower() in report_lower) for col in flagged_columns
    }
    column_recall = (
        sum(matched_columns.values()) / len(matched_columns)
        if matched_columns else 1.0  # nothing to find -> vacuously perfect
    )

    correlated_pairs = ground_truth["correlated_pairs"]
    matched_pairs = [
        pair["col1"].lower() in report_lower and pair["col2"].lower() in report_lower
        for pair in correlated_pairs
    ]
    correlation_recall = (
        sum(matched_pairs) / len(matched_pairs) if matched_pairs else 1.0
    )

    return {
        "column_recall": round(column_recall, 3),
        "matched_columns": matched_columns,
        "correlation_recall": round(correlation_recall, 3),
        "n_expected_columns": len(flagged_columns),
        "n_expected_correlations": len(correlated_pairs),
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_case(case: dict, mode: str, model: str | None, verbose: bool) -> dict:
    profile = profile_dataframe(case["df"])
    ground_truth = extract_ground_truth(profile)

    if mode == "agentic":
        from groq_agent import generate_insights_agentic
        result = generate_insights_agentic(case["df"], profile, model=model, verbose=verbose)
        report_text = result["report"]
        tool_call_count = len(result.get("tool_calls", []))
    else:
        from groq_insight_generator import generate_insights
        report_text = generate_insights(profile, chart_names=[], model=model)
        tool_call_count = 0

    scores = score_report(report_text, ground_truth)
    return {
        "case": case["name"],
        "description": case["description"],
        "mode": mode,
        "ground_truth": ground_truth,
        "scores": scores,
        "tool_call_count": tool_call_count,
        "report": report_text,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate report generators against profiler ground truth.")
    parser.add_argument("--mode", choices=["agentic", "single_shot"], default="agentic")
    parser.add_argument("--case", default=None, help="Run only this case by name (default: all cases)")
    parser.add_argument("--model", default=None, help="Override GROQ_MODEL for this run")
    parser.add_argument("--quiet", action="store_true", help="Suppress the agent's live query log")
    args = parser.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY not set — evaluate.py makes real API calls. Set it in .env and retry.", file=sys.stderr)
        sys.exit(1)

    cases = get_eval_cases()
    if args.case:
        cases = [c for c in cases if c["name"] == args.case]
        if not cases:
            print(f"No eval case named '{args.case}'. Available: {[c['name'] for c in get_eval_cases()]}", file=sys.stderr)
            sys.exit(1)

    results = []
    for case in cases:
        print(f"\n=== Running '{case['name']}' ({args.mode}) ===")
        result = run_case(case, args.mode, args.model, verbose=not args.quiet)
        results.append(result)
        s = result["scores"]
        print(
            f"  column_recall={s['column_recall']} "
            f"({sum(s['matched_columns'].values())}/{s['n_expected_columns']} flagged columns surfaced), "
            f"correlation_recall={s['correlation_recall']}"
        )
        unmatched = [c for c, ok in s["matched_columns"].items() if not ok]
        if unmatched:
            print(f"  MISSED: {unmatched}")

    avg_column_recall = sum(r["scores"]["column_recall"] for r in results) / len(results)
    avg_correlation_recall = sum(r["scores"]["correlation_recall"] for r in results) / len(results)

    print("\n=== Summary ===")
    print(f"Cases run: {len(results)}")
    print(f"Average column_recall: {round(avg_column_recall, 3)}")
    print(f"Average correlation_recall: {round(avg_correlation_recall, 3)}")

    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "mode": args.mode,
                "summary": {
                    "avg_column_recall": round(avg_column_recall, 3),
                    "avg_correlation_recall": round(avg_correlation_recall, 3),
                    "n_cases": len(results),
                },
                "results": results,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
