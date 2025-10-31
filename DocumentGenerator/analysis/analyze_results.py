#!/usr/bin/env python3
"""
Analyze metrics in a results CSV.

For each numeric metric column (e.g., exact, substring token_recall, jaccard, rouge1_f overlap, bleu, etc.):
- Compute median overall
- Compute median for rows where correct_doc_retrieved is True and False separately
- For thresholds 0.1..0.9, compute pass rates in True vs False groups (value >= threshold)
- Pick the threshold with the largest (True_pass - False_pass) difference and report it

Outputs:
- results/metric_medians.csv
- results/metric_threshold_differences.csv

Usage:
  python analyze_results.py --input best_results_best_metrics300.csv --outdir results
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import statistics
from typing import Any, Dict, List, Optional, Tuple, cast
from collections import OrderedDict
import glob
import fnmatch
from pprint import pprint


def normalize_name(name: str) -> str:
    s = name.strip().lower()
    # replace separators with underscore
    s = re.sub(r"[\s\-]+", "_", s)
    # remove any char not alnum or underscore
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s


def parse_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"true", "t", "1", "yes", "y"}:
        return True
    if s in {"false", "f", "0", "no", "n"}:
        return False
    return None


def parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return None
    # try standard float
    try:
        return float(s)
    except ValueError:
        pass
    # try replacing comma decimal
    if "," in s and s.count(",") == 1 and "." not in s:
        try:
            return float(s.replace(",", "."))
        except ValueError:
            pass
    # strip percent sign
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except ValueError:
            pass
    return None


def detect_bool_column(header: List[str]) -> Optional[str]:
    candidates = {
        "correct_doc_retrieved",
        "correct_retrieved",
        "doc_retrieved_correct",
        "retrieved_correct",
        "document_retrieved",
        "doc_retrieved",
    }
    norm_map = {normalize_name(h): h for h in header}
    for cand in candidates:
        if cand in norm_map:
            return norm_map[cand]
    # fallback: look for any column with 'correct' and 'retrieved'
    for norm, orig in norm_map.items():
        if "correct" in norm and "retrieved" in norm:
            return orig
    return None


def is_metric_like(name: str) -> bool:
    norm = normalize_name(name)
    metric_keywords = [
        "exact",
        "substring",
        "token_recall",
        "recall",
        "precision",
        "f1",
        "jaccard",
        "rouge",
        "overlap",
        "bleu",
        "meteor",
        "bert",
        "cosine",
        "similarity",
        "score",
        "accuracy",
        "acc",
    ]
    return any(k in norm for k in metric_keywords)


def load_rows(path: str) -> Tuple[List[Dict[str, Optional[str]]], List[str]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]
        header_list: List[str] = []
        if reader.fieldnames is not None:
            header_list = list(reader.fieldnames)
        elif rows:
            header_list = list(rows[0].keys())
        header = header_list
    return rows, header


def compute_prevalence(rows: List[Dict[str, Optional[str]]], bool_col: str) -> Tuple[int, int, float]:
    """Return (n_true, n_false, prevalence_true_ratio). Rows with undefined boolean are ignored."""
    n_t = 0
    n_f = 0
    for r in rows:
        b = parse_bool(r.get(bool_col))
        if b is True:
            n_t += 1
        elif b is False:
            n_f += 1
    denom = n_t + n_f
    prev = (n_t / denom) if denom > 0 else float('nan')
    return n_t, n_f, prev


def eval_file_at_thresholds(
    input_csv: str,
    metrics: List[str],
    thresholds_by_metric: Dict[str, float],
) -> Tuple[str, Dict[str, Optional[float]], Dict[str, Optional[float]], float]:
    """Evaluate at fixed thresholds for given metrics in a CSV.
    Returns (stem, {metric: J}, {metric: overall_pass_rate}, doc_retrieved_pct).
    - J = TPR - FPR at the fixed threshold (class-conditional)
    - overall_pass_rate = fraction of all non-null rows with metric >= threshold (metric score)
    If a metric is missing or has no threshold, values are None.
    """
    rows, header = load_rows(input_csv)
    stem = os.path.splitext(os.path.basename(input_csv))[0]
    bool_col = detect_bool_column(header)
    if not bool_col:
        return stem, {m: None for m in metrics}, {m: None for m in metrics}, float('nan')
    n_t, n_f, prev = compute_prevalence(rows, bool_col)

    # Map for faster lookups
    by_metric_true: Dict[str, List[Optional[float]]] = {}
    by_metric_false: Dict[str, List[Optional[float]]] = {}
    by_metric_all: Dict[str, List[Optional[float]]] = {}
    # Split once
    group_true = [r for r in rows if parse_bool(r.get(bool_col)) is True]
    group_false = [r for r in rows if parse_bool(r.get(bool_col)) is False]
    for m in metrics:
        if m not in header:
            by_metric_true[m] = []
            by_metric_false[m] = []
            by_metric_all[m] = []
            continue
        by_metric_true[m] = [parse_float(r.get(m)) for r in group_true]
        by_metric_false[m] = [parse_float(r.get(m)) for r in group_false]
        by_metric_all[m] = [parse_float(r.get(m)) for r in rows]

    out_j: Dict[str, Optional[float]] = {}
    out_score: Dict[str, Optional[float]] = {}
    for m in metrics:
        thr = thresholds_by_metric.get(m)
        tv = by_metric_true.get(m)
        fv = by_metric_false.get(m)
        av = by_metric_all.get(m)
        if thr is None or tv is None or fv is None or av is None:
            out_j[m] = None
            out_score[m] = None
            continue
        # Compute within-class pass rates for J
        t_vals = [v for v in tv if v is not None and not math.isnan(v)]
        f_vals = [v for v in fv if v is not None and not math.isnan(v)]
        a_vals = [v for v in av if v is not None and not math.isnan(v)]
        if len(t_vals) == 0 or len(f_vals) == 0 or len(a_vals) == 0:
            out_j[m] = None
            out_score[m] = None
            continue
        tpr = sum(1 for v in t_vals if v >= thr) / len(t_vals)
        fpr = sum(1 for v in f_vals if v >= thr) / len(f_vals)
        out_j[m] = tpr - fpr
        # Overall metric score at threshold: fraction of all rows passing
        overall_pass = sum(1 for v in a_vals if v >= thr) / len(a_vals)
        out_score[m] = overall_pass
    return stem, out_j, out_score, prev


def _parse_patterns(s: Optional[str]) -> List[str]:
    if not s:
        return []
    # split by comma or whitespace
    parts = re.split(r"[,\s]+", s)
    return [p.strip() for p in parts if p and p.strip()]


def _matches_any_pattern(name_candidates: List[str], patterns: List[str]) -> bool:
    if not patterns:
        return False
    # Case-insensitive: ensure candidates and patterns are lowercase
    pats = [p.lower() for p in patterns]
    for cand in name_candidates:
        c = cand.lower()
        for pat in pats:
            if fnmatch.fnmatchcase(c, pat):
                return True
    return False


def select_metric_columns(
    rows: List[Dict[str, Optional[str]]],
    header: List[str],
    bool_col: Optional[str],
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> List[str]:
    metrics: List[str] = []
    include_patterns = include_patterns or []
    exclude_patterns = exclude_patterns or []
    # If neither include nor exclude provided, keep the default behavior of skipping context*
    default_skip_context = (not include_patterns and not exclude_patterns)
    for col in header:
        if col == bool_col:
            continue
        # Explicitly skip latency-like columns
        norm_col = normalize_name(col)
        # Exclude retrieval_rank explicitly (rank is not a thresholdable quality score)
        if norm_col == 'retrieval_rank':
            continue
        # Absolute non-metric/time-based exclusions
        if any(k in norm_col for k in ["latency", "_s", "seconds"]):
            # Only skip if it clearly denotes time; allow metrics whose names just end with 's' unintentionally by stricter patterns.
            # We treat patterns like 'latency', 'latency_s', 'response_time_s'. To reduce false positives, require 'latency' or 'time' or exact suffix '_s'.
            if 'latency' in norm_col or 'time' in norm_col or norm_col.endswith('_s'):
                continue
        # Default skip for context* unless user provides include/exclude overrides
        if default_skip_context and norm_col.startswith('context'):
            continue

        # Apply user include/exclude patterns (match against original and normalized names)
        name_candidates = [col.strip().lower(), norm_col]
        if include_patterns:
            # If include is provided, only allow columns matching include
            if not _matches_any_pattern(name_candidates, include_patterns):
                continue
        # Exclude has priority after include check
        if exclude_patterns and _matches_any_pattern(name_candidates, exclude_patterns):
            continue
        # Consider numeric columns as metrics, with a bias to metric-like names.
        numeric_count = 0
        total_count = 0
        for r in rows:
            v = r.get(col)
            x = parse_float(v)
            if v is None or str(v).strip() == "":
                continue
            total_count += 1
            if x is not None and not math.isnan(x):
                numeric_count += 1
        if numeric_count == 0:
            continue
        # If most non-empty values are numeric, accept.
        if total_count == 0 or (numeric_count / max(total_count, 1) >= 0.6) or is_metric_like(col):
            metrics.append(col)
    return metrics


def median_safe(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return None
    try:
        return statistics.median(vals)
    except statistics.StatisticsError:
        return None


def compute_pass_rate(values: List[Optional[float]], threshold: float, group_size: int) -> Tuple[float, int]:
    """Return pass rate within this group (TPR/FPR style) and the group size used.
    Rate = passed_in_group / group_size, ignoring None/NaN values.
    """
    vals = [v for v in values if v is not None and not math.isnan(v)]
    n = len(vals)
    if n == 0:
        return 0.0, 0
    passed = sum(1 for v in vals if v >= threshold)
    return passed / group_size, n


def wilson_interval(p: float, n: int, z: float = 1.96) -> Tuple[Optional[float], Optional[float]]:
    if n == 0:
        return None, None
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def compute_auc(values: List[Optional[float]], labels: List[Optional[bool]]) -> Optional[float]:
    pairs = [(v, l) for v, l in zip(values, labels) if v is not None and l is not None]
    if not pairs:
        return None
    pos = [v for v, l in pairs if l]
    neg = [v for v, l in pairs if not l]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return None
    # rank with average ties
    sorted_vals = sorted((v, i) for i, (v, _) in enumerate(pairs))
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1][0] == sorted_vals[i][0]:
            j += 1
        avg_rank = (i + j + 2) / 2
        for k in range(i, j + 1):
            ranks[sorted_vals[k][1]] = avg_rank
        i = j + 1
    pos_rank_sum = sum(r for r, (_, l) in zip(ranks, pairs) if l)
    u = pos_rank_sum - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def permutation_p_value(values: List[Optional[float]], labels: List[Optional[bool]], thresholds: List[float], observed_j: float, perm_n: int, rng: random.Random) -> Optional[float]:
    clean = [(v, l) for v, l in zip(values, labels) if v is not None and l is not None]
    if not clean or perm_n <= 0:
        return None
    vals = [v for v, _ in clean]
    labs = [l for _, l in clean]
    n_pos = sum(1 for l in labs if l)
    n_neg = len(labs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    count_ge = 0
    for _ in range(perm_n):
        rng.shuffle(labs)
        pos_vals = [vals[i] for i, lb in enumerate(labs) if lb]
        neg_vals = [vals[i] for i, lb in enumerate(labs) if not lb]
        best_j = -1.0
        for t in thresholds:
            if pos_vals:
                tpr = sum(1 for v in pos_vals if v >= t) / len(pos_vals)
            else:
                tpr = 0.0
            if neg_vals:
                fpr = sum(1 for v in neg_vals if v >= t) / len(neg_vals)
            else:
                fpr = 0.0
            j_val = tpr - fpr
            if j_val > best_j:
                best_j = j_val
        if best_j >= observed_j - 1e-12:
            count_ge += 1
    return (count_ge + 1) / (perm_n + 1)


def analyze_file(
    input_csv: str,
    outdir: str,
    thresholds: List[float] | None = None,
    include_threshold_one: bool = True,
    perm_n: int = 0,
    rng: Optional[random.Random] = None,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if thresholds is None:
        thresholds = [round(x / 100.0, 2) for x in range(1, 100)]  # 0.01..0.99 by 0.01
    if include_threshold_one and 1.0 not in thresholds:
        thresholds = sorted(set(thresholds + [1.0]))
    if rng is None:
        rng = random.Random(0)

    rows, header = load_rows(input_csv)
    if not rows:
        raise SystemExit("No rows found in input CSV")

    bool_col = detect_bool_column(header)
    if not bool_col:
        raise SystemExit(
            "Could not find 'correct_doc_retrieved' column (or similar). Please ensure the CSV has it."
        )

    metric_cols = select_metric_columns(rows, header, bool_col, include_patterns=include_patterns, exclude_patterns=exclude_patterns)
    if not metric_cols:
        raise SystemExit("No metric columns detected.")

    # Prepare groupings
    group_true: List[Dict[str, Optional[str]]] = []
    group_false: List[Dict[str, Optional[str]]] = []
    for r in rows:
        b = parse_bool(r.get(bool_col))
        if b is True:
            group_true.append(r)
        elif b is False:
            group_false.append(r)
        else:
            # skip rows with undefined boolean
            pass

    # Build results
    medians_summary: List[Dict[str, Any]] = []
    thresholds_summary: List[Dict[str, Any]] = []

    for col in metric_cols:
        # collect values
        all_vals: List[Optional[float]] = [parse_float(r.get(col)) for r in rows]
        true_vals: List[Optional[float]] = [parse_float(r.get(col)) for r in group_true]
        false_vals: List[Optional[float]] = [parse_float(r.get(col)) for r in group_false]
        labels_all: List[Optional[bool]] = [parse_bool(r.get(bool_col)) for r in rows]

        med_overall = median_safe([v for v in all_vals if v is not None])
        med_true = median_safe([v for v in true_vals if v is not None])
        med_false = median_safe([v for v in false_vals if v is not None])

        medians_summary.append(
            {
                "metric": col,
                "median_overall": med_overall,
                "median_true": med_true,
                "median_false": med_false,
                "n_overall": sum(1 for v in all_vals if v is not None),
                "n_true": sum(1 for v in true_vals if v is not None),
                "n_false": sum(1 for v in false_vals if v is not None),
            }
        )

        # Threshold scan
        median_gap = (med_true - med_false) if (med_true is not None and med_false is not None) else None
        best: Dict[str, Any] = {
            "metric": col,
            "best_threshold": None,
            "true_pass_rate": None,
            "false_pass_rate": None,
            "difference": None,
            "tpr_ci_low": None,
            "tpr_ci_high": None,
            "fpr_ci_low": None,
            "fpr_ci_high": None,
            "auc": None,
            "p_value": None,
            "median_gap": median_gap,
            "n_true": sum(1 for v in true_vals if v is not None),
            "n_false": sum(1 for v in false_vals if v is not None),
            # Class balance ratio in [0,1]: 1 means perfectly balanced n_true == n_false
            "class_balance_ratio": None,
            "median_overall": med_overall,
            "median_true": med_true,
            "median_false": med_false,
        }

        max_diff = -float("inf")
        best_tuple = None
        n_true_nonnull = sum(1 for v in true_vals if v is not None)
        n_false_nonnull = sum(1 for v in false_vals if v is not None)
        for t in thresholds:
            pr_true, _ = compute_pass_rate(true_vals, t, len(all_vals))
            pr_false, _ = compute_pass_rate(false_vals, t, len(all_vals))
            diff = pr_true - pr_false
            tup = (diff, t, pr_true, pr_false)
            if diff > max_diff:
                max_diff = diff
                best_tuple = tup
        if best_tuple is not None:
            diff, t, pr_true, pr_false = best_tuple
            tpr_ci_low, tpr_ci_high = wilson_interval(pr_true, n_true_nonnull)
            fpr_ci_low, fpr_ci_high = wilson_interval(pr_false, n_false_nonnull)
            auc_val = compute_auc(all_vals, labels_all)
            p_val = permutation_p_value(all_vals, labels_all, thresholds, diff, perm_n, rng) if perm_n > 0 else None
            # overall pass rate across ALL rows at the chosen threshold (ignoring labels)
            all_nonnull = [v for v in all_vals if v is not None and not math.isnan(v)]
            if len(all_nonnull) > 0:
                overall_pass_rate = sum(1 for v in all_nonnull if v >= t) / len(all_nonnull)
            else:
                overall_pass_rate = None
            # compute class balance ratio
            if (n_true_nonnull or 0) > 0 or (n_false_nonnull or 0) > 0:
                denom = max(n_true_nonnull, n_false_nonnull)
                numer = min(n_true_nonnull, n_false_nonnull)
                balance_ratio = (numer / denom) if denom > 0 else None
            else:
                balance_ratio = None
            # percent of documents retrieved (i.e., positive prevalence among non-null)
            if (n_true_nonnull + n_false_nonnull) > 0:
                doc_retrieved_pct = n_true_nonnull / (n_true_nonnull + n_false_nonnull)
            else:
                doc_retrieved_pct = None
            best.update(
                {
                    "best_threshold": t,
                    "true_pass_rate": pr_true,
                    "false_pass_rate": pr_false,
                    "difference": diff,
                    # For clarity downstream, keep the direct overall pass rate too
                    "overall_pass_rate": overall_pass_rate,
                    "tpr_ci_low": tpr_ci_low,
                    "tpr_ci_high": tpr_ci_high,
                    "fpr_ci_low": fpr_ci_low,
                    "fpr_ci_high": fpr_ci_high,
                    "auc": auc_val,
                    "p_value": p_val,
                    "class_balance_ratio": balance_ratio,
                    "doc_retrieved_pct": doc_retrieved_pct,
                }
            )

        thresholds_summary.append(best)

    # Write outputs (prefix with input file stem)
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(input_csv))[0]
    medians_path = os.path.join(outdir, f"{stem}_metric_medians.csv")
    thresholds_path = os.path.join(outdir, f"{stem}_metric_threshold_differences.csv")

    if medians_summary:
        with open(medians_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "metric",
                    "median_overall",
                    "median_true",
                    "median_false",
                    "n_overall",
                    "n_true",
                    "n_false",
                ],
            )
            writer.writeheader()
            writer.writerows(medians_summary)

    if thresholds_summary:
        with open(thresholds_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "metric",
                    "best_threshold",
                    "true_pass_rate",
                    "false_pass_rate",
                    "difference",
                    "tpr_ci_low",
                    "tpr_ci_high",
                    "fpr_ci_low",
                    "fpr_ci_high",
                    "auc",
                    "p_value",
                    "median_gap",
                    "class_balance_ratio",
                    "doc_retrieved_pct",
                    "n_true",
                    "n_false",
                    "median_overall",
                    "median_true",
                    "median_false",
                ],
            )
            writer.writeheader()
            cleaned_rows = []
            for r in thresholds_summary:
                cleaned_rows.append({k: r.get(k) for k in writer.fieldnames})
            writer.writerows(cleaned_rows)

    return medians_summary, thresholds_summary


def format_pct(x: Optional[float]) -> str:
    if x is None or math.isnan(x):
        return "NA"
    return f"{x*100:.1f}%"


def format_pct3(x: Optional[float]) -> str:
    """Compact percentage for console: aim for ~3 characters like '45%'.
    Returns '100%' for 1.0.
    """
    if x is None or math.isnan(x):
        return "NA"
    val = int(round(x * 100))
    # Keep '100%' for edge case, otherwise 0..99%
    return f"{val}%"


def should_format_as_pct(metric_name: str, *vals: Optional[float]) -> bool:
    name = normalize_name(metric_name)
    # Exclude time/latency/rank-like metrics
    if any(k in name for k in ["latency", "time", "sec", "rank"]):
        return False
    present = [v for v in vals if v is not None and not math.isnan(v)]
    if not present:
        return False
    # If all present medians are in [0,1], treat as percentage
    return all(0.0 <= v <= 1.0 for v in present)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze results CSV for metric medians and thresholds")
    parser.add_argument(
        "--input", "-i",
        nargs='+',
        default=["best_results_best_metrics300.csv"],
        help="Path(s) to input CSV(s). Provide one or more."
    )
    parser.add_argument("--outdir", "-o", default="results", help="Directory to write outputs")
    parser.add_argument("--perm-n", type=int, default=0, help="Number of permutations for p-value (0 disables)")
    parser.add_argument("--select-by", choices=["difference", "auc", "median_gap"], default="difference", help="Criterion for selecting top-K metrics per file")
    parser.add_argument("--top-k", type=int, default=15, help="Top K metrics per file for threshold report")
    parser.add_argument("--json-summary", default=None, help="Optional path to write JSON summary (default: outdir/summary.json)")
    parser.add_argument("--min-balance", type=float, default=0, help="Minimum class balance ratio required (min(n_true,n_false)/max(n_true,n_false)); range [0,1]. Rows below are excluded from top-K selection")
    parser.add_argument(
        "--exclude",
        default=None,
        help="Comma or space separated glob patterns to exclude metrics by name (matches original and normalized). Example: context*, debug_*",
    )
    parser.add_argument(
        "--include",
        default=None,
        help="Comma or space separated glob patterns to include. When provided, only matching metrics are considered (after built-in non-metric/time exclusions).",
    )
    parser.add_argument(
        "--fixed-threshold-from",
        default="AgentOpenAI_300",
        help="File stem to source per-metric thresholds from (e.g., AgentOpenAI_300). Use its best J threshold per metric.",
    )
    parser.add_argument(
        "--emit-fixed-threshold-tables",
        action="store_true",
        help="When set, computes a cross-file table of J evaluated at thresholds from --fixed-threshold-from and saves CSVs.",
    )
    args = parser.parse_args()

    # Clamp min-balance to [0,1]
    if args.min_balance is None or not isinstance(args.min_balance, (int, float)):
        args.min_balance = 0
    else:
        args.min_balance = max(0.0, min(1.0, float(args.min_balance)))

    # Support multiple input files or directories
    raw_inputs: List[str] = list(args.input)
    include_patterns = _parse_patterns(args.include)
    exclude_patterns = _parse_patterns(args.exclude)

    def is_output_csv(filename: str) -> bool:
        fname = os.path.basename(filename).lower()
        return (
            fname.endswith("_metric_medians.csv")
            or fname.endswith("_metric_threshold_differences.csv")
            or fname == "final_top_thresholds.csv"
        )

    expanded: List[str] = []
    for p in raw_inputs:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                if not name.lower().endswith(".csv"):
                    continue
                full = os.path.join(p, name)
                if os.path.isfile(full) and not is_output_csv(full):
                    expanded.append(full)
        elif os.path.isfile(p):
            if p.lower().endswith(".csv") and not is_output_csv(p):
                expanded.append(p)
        else:
            # glob pattern
            for g in sorted(glob.glob(p)):
                if os.path.isfile(g) and g.lower().endswith(".csv") and not is_output_csv(g):
                    expanded.append(g)

    # Deduplicate while preserving order
    input_files: List[str] = list(OrderedDict.fromkeys(expanded).keys())
    if not input_files:
        raise SystemExit("No CSV files found from the provided --input arguments.")
    all_results: List[Dict[str, Any]] = []
    per_file_summaries: List[Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]] = []

    rng = random.Random(0)
    for in_path in input_files:
        medians_summary, thresholds_summary = analyze_file(
            in_path,
            args.outdir,
            perm_n=args.perm_n,
            rng=rng,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
        # attach file source to threshold rows
        stem = os.path.splitext(os.path.basename(in_path))[0]
        for row in thresholds_summary:
            row_with_src = dict(row)
            row_with_src["source_file"] = stem
            all_results.append(row_with_src)
        per_file_summaries.append((stem, medians_summary, thresholds_summary))

    # Console output per file
    for stem, medians_summary, thresholds_summary in per_file_summaries:
        print(f"\n[File: {stem}] Metric medians (overall / True / False):")
        for row in medians_summary:
            mo = cast(Optional[float], row.get('median_overall'))
            mt = cast(Optional[float], row.get('median_true'))
            mf = cast(Optional[float], row.get('median_false'))
            metric_name = cast(str, row.get('metric'))
            # compute balance from counts
            try:
                n_t = int(cast(Any, row.get('n_true')))
                n_f = int(cast(Any, row.get('n_false')))
            except Exception:
                n_t, n_f = 0, 0
            if max(n_t, n_f) > 0:
                balance_ratio_med = min(n_t, n_f) / max(n_t, n_f)
            else:
                balance_ratio_med = None
            if should_format_as_pct(metric_name, mo, mt, mf):
                mo_s = format_pct3(mo)
                mt_s = format_pct3(mt)
                mf_s = format_pct3(mf)
            else:
                mo_s = "NA" if mo is None else f"{mo}"
                mt_s = "NA" if mt is None else f"{mt}"
                mf_s = "NA" if mf is None else f"{mf}"
            print(
                f"- {metric_name}: {mo_s} / {mt_s} / {mf_s} | Balance={format_pct(balance_ratio_med)} "
                f"(n: {row['n_overall']}/{row['n_true']}/{row['n_false']})"
            )

    # Per-file best thresholds by metric (ordered by highest threshold) for each file
    def _best_threshold_key(r: Dict[str, Any]) -> float:
        v = r.get('best_threshold')
        if isinstance(v, (int, float)):
            return float(v)
        try:
            if v is not None:
                return float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass
        return float('-inf')

    for stem, _, thresholds_summary in per_file_summaries:
        print(f"\n[File: {stem}] Best thresholds by metric (max True-False pass-rate difference):")
        sorted_thresholds = sorted(
            thresholds_summary,
            key=_best_threshold_key,
            reverse=True,
        )
        for row in sorted_thresholds:
            true_pass = cast(Optional[float], row.get('true_pass_rate'))
            false_pass = cast(Optional[float], row.get('false_pass_rate'))
            diff_pass = cast(Optional[float], row.get('difference'))
            balance = cast(Optional[float], row.get('class_balance_ratio'))
            doc_pct = cast(Optional[float], row.get('doc_retrieved_pct'))
            print(
                f"- {row['metric']}: threshold={row['best_threshold']} | "
                f"TPR={format_pct(true_pass)} | FPR={format_pct(false_pass)} | "
                f"J={format_pct(diff_pass)} | Doc%={format_pct(doc_pct)} | Bal={format_pct(balance)}"
            )

    # Build final combined report: top 3 thresholds per input file, ranked overall by highest threshold
    def _diff_key(r: Dict[str, Any]) -> float:
        v = r.get('difference')
        if isinstance(v, (int, float)):
            return float(v)
        try:
            if v is not None:
                return float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass
        return float('-inf')

    TOP_K_PER_FILE = args.top_k
    final_rows: List[Dict[str, Any]] = []
    select_mode = args.select_by
    def _auc_key(r: Dict[str, Any]) -> float:
        v = r.get('auc')
        return float(v) if isinstance(v, (int, float)) else -float('inf')
    def _median_gap_key(r: Dict[str, Any]) -> float:
        v = r.get('median_gap')
        return float(v) if isinstance(v, (int, float)) else -float('inf')
    def _balance_key(r: Dict[str, Any]) -> float:
        v = r.get('class_balance_ratio')
        return float(v) if isinstance(v, (int, float)) else -float('inf')
    for stem, _, thresholds_summary in per_file_summaries:
        # Hard constraint: filter by class balance ratio
        filtered = []
        for r in thresholds_summary:
            v = r.get('class_balance_ratio')
            if isinstance(v, (int, float)) and float(v) >= args.min_balance:
                filtered.append(r)
        if not filtered:
            print(f"\n[Note] Skipping file '{stem}' in final top thresholds: no metrics meet --min-balance={args.min_balance}")
            continue
        # Primary sort by chosen criterion; tie-break by class balance ratio (descending), then by threshold (descending)
        if select_mode == 'difference':
            sorted_sel = sorted(
                filtered,
                key=lambda r: (_diff_key(r), _balance_key(r), _best_threshold_key(r)),
                reverse=True,
            )
        elif select_mode == 'auc':
            sorted_sel = sorted(
                filtered,
                key=lambda r: (_auc_key(r), _balance_key(r), _best_threshold_key(r)),
                reverse=True,
            )
        elif select_mode == 'median_gap':
            sorted_sel = sorted(
                filtered,
                key=lambda r: (_median_gap_key(r), _balance_key(r), _best_threshold_key(r)),
                reverse=True,
            )
        else:
            sorted_sel = sorted(
                filtered,
                key=lambda r: (_diff_key(r), _balance_key(r), _best_threshold_key(r)),
                reverse=True,
            )
        topk = sorted_sel[:TOP_K_PER_FILE]
        for r in topk:
            rr = dict(r)
            rr['source_file'] = stem
            # Prefer the directly computed overall pass rate (from all values) if present
            overall_direct = rr.get('overall_pass_rate')
            if isinstance(overall_direct, (int, float)):
                rr['avg_pass_rate'] = float(overall_direct)
            else:
                # Fallback: compute from TPR/FPR weighted by class counts
                tpr = rr.get('true_pass_rate')
                fpr = rr.get('false_pass_rate')
                n_t = rr.get('n_true')
                n_f = rr.get('n_false')
                overall_pass: Optional[float]
                if (
                    isinstance(tpr, (int, float))
                    and isinstance(fpr, (int, float))
                    and isinstance(n_t, (int, float))
                    and isinstance(n_f, (int, float))
                ):
                    n_t_f = float(n_t)
                    n_f_f = float(n_f)
                    denom = n_t_f + n_f_f
                    if denom > 0:
                        overall_pass = (float(tpr) * n_t_f + float(fpr) * n_f_f) / denom
                    else:
                        overall_pass = None
                else:
                    overall_pass = None
                rr['avg_pass_rate'] = overall_pass
            final_rows.append(rr)

    # sort combined final rows by threshold desc (primary report)
    final_rows_sorted = sorted(final_rows, key=_best_threshold_key, reverse=True)

    # Average pass rate key for alternative ordering (internal use only now)
    def _row_id(r: Dict[str, Any]) -> str:
        return f"{r.get('source_file')}|{r.get('metric')}|{r.get('best_threshold')}"

    def _avg_key(r: Dict[str, Any]) -> float:
        v = r.get('avg_pass_rate')
        if isinstance(v, (int, float)):
            return float(v)
        try:
            if v is not None:
                return float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass
        return float('-inf')

    print(f"\nFinal combined top thresholds (top {TOP_K_PER_FILE} per file, ranked by threshold; selection criterion={select_mode}):")
    for row in final_rows_sorted:
        true_pass = cast(Optional[float], row.get('true_pass_rate'))
        false_pass = cast(Optional[float], row.get('false_pass_rate'))
        diff_pass = cast(Optional[float], row.get('difference'))
        balance = cast(Optional[float], row.get('class_balance_ratio'))
        doc_pct = cast(Optional[float], row.get('doc_retrieved_pct'))
        avg_pass = cast(Optional[float], row.get('avg_pass_rate'))
        print(
            f"- [File: {row['source_file']}] {row['metric']}: threshold={row['best_threshold']} | "
            f"TPR={format_pct(true_pass)} | FPR={format_pct(false_pass)} | J={format_pct(diff_pass)} | "
            f"Doc%={format_pct(doc_pct)} | Balance={format_pct(balance)} | Overall={format_pct(avg_pass)}"
        )

    # Build and print a combined list ranked by median_overall: top 3 medians per file
    TOPK_MEDS = TOP_K_PER_FILE
    combined_medians: List[Dict[str, Any]] = []
    for stem, medians_summary, _ in per_file_summaries:
        # sort medians by median_overall desc, filter None
        def _med_overall_key(r: Dict[str, Any]) -> float:
            v = r.get('median_overall')
            if isinstance(v, (int, float)):
                return float(v)
            try:
                if v is not None:
                    return float(v)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                pass
            return float('-inf')

        sorted_meds = sorted(medians_summary, key=_med_overall_key, reverse=True)
        topk_meds = []
        for m in sorted_meds:
            mv = m.get('median_overall')
            if mv is None or (isinstance(mv, float) and math.isnan(mv)):
                continue
            topk_meds.append(m)
            if len(topk_meds) >= TOPK_MEDS:
                break
        for m in topk_meds:
            rr = dict(m)
            rr['source_file'] = stem
            combined_medians.append(rr)

    # sort combined by median_overall desc
    def _med_overall_key2(r: Dict[str, Any]) -> float:
        v = r.get('median_overall')
        if isinstance(v, (int, float)):
            return float(v)
        try:
            if v is not None:
                return float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass
        return float('-inf')

    combined_medians_sorted = sorted(combined_medians, key=_med_overall_key2, reverse=True)

    print("\nFinal combined top medians (top 3 per file, ranked by median_overall):")
    for row in combined_medians_sorted:
        metric_name = cast(str, row.get('metric'))
        mo = cast(Optional[float], row.get('median_overall'))
        mt = cast(Optional[float], row.get('median_true'))
        mf = cast(Optional[float], row.get('median_false'))
        # compute balance from counts
        try:
            n_t = int(cast(Any, row.get('n_true')))
            n_f = int(cast(Any, row.get('n_false')))
        except Exception:
            n_t, n_f = 0, 0
        if max(n_t, n_f) > 0:
            balance_ratio_med = min(n_t, n_f) / max(n_t, n_f)
        else:
            balance_ratio_med = None
        if should_format_as_pct(metric_name, mo, mt, mf):
            mo_s = format_pct3(mo)
            mt_s = format_pct3(mt)
            mf_s = format_pct3(mf)
        else:
            mo_s = "NA" if mo is None else f"{mo}"
            mt_s = "NA" if mt is None else f"{mt}"
            mf_s = "NA" if mf is None else f"{mf}"
        print(
            f"- [File: {row['source_file']}] {metric_name}: median_overall={mo_s} | median_true={mt_s} | median_false={mf_s} | Balance={format_pct(balance_ratio_med)} "
            f"(n: {row['n_overall']}/{row['n_true']}/{row['n_false']})"
        )

    # Save combined medians to CSV
    combined_medians_path = os.path.join(args.outdir, "final_top_medians.csv")
    if combined_medians_sorted:
        with open(combined_medians_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "source_file",
                    "metric",
                    "median_overall",
                    "median_true",
                    "median_false",
                    "n_overall",
                    "n_true",
                    "n_false",
                ],
            )
            writer.writeheader()
            writer.writerows(combined_medians_sorted)

    # Save final combined report
    os.makedirs(args.outdir, exist_ok=True)
    final_path = os.path.join(args.outdir, "final_top_thresholds.csv")
    if final_rows_sorted:
        with open(final_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "source_file",
                    "metric",
                    "best_threshold",
                    "true_pass_rate",
                    "false_pass_rate",
                    "difference",
                    "tpr_ci_low",
                    "tpr_ci_high",
                    "fpr_ci_low",
                    "fpr_ci_high",
                    "auc",
                    "p_value",
                    "median_gap",
                    "class_balance_ratio",
                    "doc_retrieved_pct",
                    "overall_pass_rate",
                    "avg_pass_rate",
                    "n_true",
                    "n_false",
                    "median_overall",
                    "median_true",
                    "median_false",
                ],
            )
            writer.writeheader()
            writer.writerows(final_rows_sorted)

    # Optional: emit cross-file fixed-threshold J tables using thresholds from a reference file
    if args.emit_fixed_threshold_tables:
        ref = args.fixed_threshold_from
        # Build per-metric threshold dict from final_rows_sorted for the reference file
        ref_rows = [r for r in final_rows_sorted if str(r.get('source_file')) == ref]
        thresholds_by_metric: Dict[str, float] = {}
        for r in ref_rows:
            m = str(r.get('metric'))
            t = r.get('best_threshold')
            if isinstance(t, (int, float)):
                thresholds_by_metric[m] = float(t)
        # Define column order
        metric_cols_order = [
            'document_retrieved',
            'token_recall',
            'jaccard',
            'rouge1_f',
            'overlap',
            'bleu',
            'bert_cos',
        ]
        # Evaluate per file
        fixed_rows_j: List[Dict[str, Any]] = []
        fixed_rows_score: List[Dict[str, Any]] = []
        for p in input_files:
            stem = os.path.splitext(os.path.basename(p))[0]
            # Use presence of columns to decide which metrics to compute
            metrics_present = [m for m in metric_cols_order[1:] if any(m == k for k in thresholds_by_metric.keys())]
            s, j_map, score_map, prev = eval_file_at_thresholds(p, metrics_present, thresholds_by_metric)
            row_j = {'file': s, 'document_retrieved': prev}
            row_score = {'file': s, 'document_retrieved': prev}
            for m in metric_cols_order[1:]:
                row_j[m] = j_map.get(m)
                row_score[m] = score_map.get(m)
            fixed_rows_j.append(row_j)
            fixed_rows_score.append(row_score)
        # Write CSVs (J and score)
        fixed_path_j = os.path.join(args.outdir, 'fixed_threshold_j_by_reference.csv')
        with open(fixed_path_j, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['file'] + metric_cols_order)
            writer.writeheader()
            writer.writerows(fixed_rows_j)
        print(f"\nSaved fixed-threshold J table (reference={ref}): {fixed_path_j}")

        fixed_path_score = os.path.join(args.outdir, 'fixed_threshold_score_by_reference.csv')
        with open(fixed_path_score, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['file'] + metric_cols_order)
            writer.writeheader()
            writer.writerows(fixed_rows_score)
        print(f"Saved fixed-threshold score table (overall pass rate at reference thresholds): {fixed_path_score}")

        # Derive Accuracy@Tref and BalancedAccuracy@Tref per metric using: 
        # given prevalence p, overall score s, and J = tpr - fpr
        # fpr = s - p * J, tpr = s + (1 - p) * J
        # accuracy = p * tpr + (1 - p) * (1 - fpr)
        # balanced_accuracy = (tpr + (1 - fpr)) / 2
        fixed_rows_acc: List[Dict[str, Any]] = []
        fixed_rows_balacc: List[Dict[str, Any]] = []
        # Build quick maps for j and score by file for convenience
        j_by_file = {row['file']: row for row in fixed_rows_j}
        s_by_file = {row['file']: row for row in fixed_rows_score}
        for p in input_files:
            stem = os.path.splitext(os.path.basename(p))[0]
            jrow = j_by_file.get(stem)
            srow = s_by_file.get(stem)
            if not jrow or not srow:
                continue
            p_prev = jrow.get('document_retrieved')
            acc_row: Dict[str, Any] = {'file': stem, 'document_retrieved': p_prev}
            balacc_row: Dict[str, Any] = {'file': stem, 'document_retrieved': p_prev}
            for m in metric_cols_order[1:]:
                jv = jrow.get(m)
                sv = srow.get(m)
                if (
                    isinstance(p_prev, (int, float)) and not math.isnan(p_prev)
                    and isinstance(jv, (int, float)) and not math.isnan(jv)
                    and isinstance(sv, (int, float)) and not math.isnan(sv)
                ):
                    p_val = float(p_prev)
                    fpr = sv - p_val * jv
                    tpr = sv + (1.0 - p_val) * jv
                    # Clamp to [0,1] to reduce numeric drift
                    fpr = min(1.0, max(0.0, fpr))
                    tpr = min(1.0, max(0.0, tpr))
                    acc = p_val * tpr + (1.0 - p_val) * (1.0 - fpr)
                    balacc = 0.5 * (tpr + (1.0 - fpr))
                    acc_row[m] = acc
                    balacc_row[m] = balacc
                else:
                    acc_row[m] = None
                    balacc_row[m] = None
            fixed_rows_acc.append(acc_row)
            fixed_rows_balacc.append(balacc_row)

        fixed_path_acc = os.path.join(args.outdir, 'fixed_threshold_accuracy_by_reference.csv')
        with open(fixed_path_acc, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['file'] + metric_cols_order)
            writer.writeheader()
            writer.writerows(fixed_rows_acc)
        print(f"Saved fixed-threshold accuracy table (at reference thresholds): {fixed_path_acc}")

        fixed_path_balacc = os.path.join(args.outdir, 'fixed_threshold_balanced_accuracy_by_reference.csv')
        with open(fixed_path_balacc, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['file'] + metric_cols_order)
            writer.writeheader()
            writer.writerows(fixed_rows_balacc)
        print(f"Saved fixed-threshold balanced accuracy table (at reference thresholds): {fixed_path_balacc}")

        # Console previews
        def _fmt3(v: Any) -> str:
            if isinstance(v, (int, float)) and not math.isnan(v):
                return f"{v:.3f}"
            if v is None:
                return "NA"
            return str(v)

        print("\nFixed-threshold J (at reference thresholds):")
        header_cols = ['file'] + metric_cols_order
        print("\t".join(header_cols))
        for r in fixed_rows_j:
            print("\t".join(_fmt3(r.get(c)) for c in header_cols))

        print("\nFixed-threshold metric score (overall pass rate at reference thresholds):")
        print("\t".join(header_cols))
        for r in fixed_rows_score:
            print("\t".join(_fmt3(r.get(c)) for c in header_cols))

        print("\nFixed-threshold Accuracy (at reference thresholds):")
        print("\t".join(header_cols))
        for r in fixed_rows_acc:
            print("\t".join(_fmt3(r.get(c)) for c in header_cols))

        print("\nFixed-threshold Balanced Accuracy (at reference thresholds):")
        print("\t".join(header_cols))
        for r in fixed_rows_balacc:
            print("\t".join(_fmt3(r.get(c)) for c in header_cols))

        # Also emit a median control table for the same columns
        med_control: List[Dict[str, Any]] = []
        # Build a quick lookup for medians by (file, metric)
        med_lookup: Dict[Tuple[str,str], Dict[str, Any]] = {}
        for stem, medians_summary, _ in per_file_summaries:
            for m in medians_summary:
                med_lookup[(stem, str(m.get('metric')))] = m
        for stem, _, _ in per_file_summaries:
            row = {'file': stem, 'document_retrieved': None}
            for m in metric_cols_order[1:]:
                mm = med_lookup.get((stem, m))
                if mm:
                    row[m] = mm.get('median_overall')
                else:
                    row[m] = None
            med_control.append(row)
        med_path = os.path.join(args.outdir, 'median_control_by_file.csv')
        with open(med_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['file'] + metric_cols_order)
            writer.writeheader()
            writer.writerows(med_control)
        print(f"Saved median control table: {med_path}")
        # Console preview
        print("\nMedian control (median_overall per metric):")
        print("\t".join(['file'] + metric_cols_order))
        for r in med_control:
            def _fmt2(v):
                if isinstance(v, (int, float)) and not math.isnan(v):
                    return f"{v:.3f}"
                if v is None:
                    return "NA"
                return str(v)
            print("\t".join(_fmt2(r.get(c)) for c in ['file'] + metric_cols_order))

    summary_path = args.json_summary or os.path.join(args.outdir, 'summary.json')
    try:
        summary_obj = {
            'selection_criterion': select_mode,
            'top_k': TOP_K_PER_FILE,
            'min_balance': args.min_balance,
            'files_processed': input_files,
            'final_threshold_rows': final_rows_sorted,
            'final_median_rows': combined_medians_sorted,
        }
        with open(summary_path, 'w', encoding='utf-8') as jf:
            json.dump(summary_obj, jf, indent=2, default=str)
    except Exception as e:
        print(f"Warning: could not write JSON summary ({e})")

    print("\nSaved:")
    # List per-file outputs
    for stem, _, _ in per_file_summaries:
        print(f"- {os.path.join(args.outdir, stem + '_metric_medians.csv')}")
        print(f"- {os.path.join(args.outdir, stem + '_metric_threshold_differences.csv')}")
    print(f"- {os.path.join(args.outdir, 'final_top_thresholds.csv')}")
    print(f"- {os.path.join(args.outdir, 'final_top_medians.csv')}")
    print(f"- {summary_path}")


if __name__ == "__main__":
    main()
