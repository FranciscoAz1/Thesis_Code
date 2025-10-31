#!/usr/bin/env python3
"""
Split a CSV into two files based on whether a given column ends with .docx.

Defaults:
- input: merged_data/AgentOpenAI_Mixed.csv
- column: expected_file (case-insensitive match against header)
- outputs (written next to input file):
  - Agent_OpenAI_mixeddocx.csv (rows where value endswith .docx)
  - Agent_OpenAI_mixedRest.csv (all other rows)

Usage examples:
  python split_by_extension.py
  python split_by_extension.py --input merged_data/AgentOpenAI_Mixed.csv --column expected_file \
    --docx-out Agent_OpenAI_mixeddocx.csv --rest-out Agent_OpenAI_mixedRest.csv
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
from typing import Optional, Sequence


def detect_column(fieldnames: Sequence[str], target: str) -> Optional[str]:
    target_norm = target.strip().lower()
    for name in fieldnames:
        if name is None:
            continue
        if name.strip().lower() == target_norm:
            return name
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Split CSV by .docx extension in a specified column")
    parser.add_argument("--input", "-i", default=os.path.join("merged_data", "AgentOpenAI_Mixed.csv"), help="Path to input CSV")
    parser.add_argument("--column", "-c", default="expected_file", help="Column name to inspect (case-insensitive match)")
    parser.add_argument("--docx-out", default="Agent_OpenAI_mixeddocx.csv", help="Output filename for .docx rows (written next to input if relative)")
    parser.add_argument("--rest-out", default="Agent_OpenAI_mixedRest.csv", help="Output filename for non-.docx rows (written next to input if relative)")
    args = parser.parse_args(argv)

    in_path = args.input
    if not os.path.isfile(in_path):
        print(f"ERROR: Input file not found: {in_path}", file=sys.stderr)
        return 2

    # Resolve output paths relative to input directory if not absolute
    in_dir = os.path.dirname(os.path.abspath(in_path))
    docx_out = args.docx_out if os.path.isabs(args.docx_out) else os.path.join(in_dir, args.docx_out)
    rest_out = args.rest_out if os.path.isabs(args.rest_out) else os.path.join(in_dir, args.rest_out)

    os.makedirs(os.path.dirname(docx_out), exist_ok=True)
    os.makedirs(os.path.dirname(rest_out), exist_ok=True)

    total = 0
    docx_count = 0
    rest_count = 0

    # Read and write with UTF-8, handle BOM on input
    with open(in_path, "r", encoding="utf-8-sig", newline="") as f_in:
        reader = csv.DictReader(f_in)
        if not reader.fieldnames:
            print("ERROR: No header row found in CSV.", file=sys.stderr)
            return 3
        col = detect_column(reader.fieldnames, args.column)
        if not col:
            print(
                "ERROR: Column not found (case-insensitive):",
                args.column,
                "\nAvailable columns:",
                ", ".join(reader.fieldnames),
                file=sys.stderr,
            )
            return 4

        # Prepare writers with same fieldnames to preserve column order
        with open(docx_out, "w", encoding="utf-8", newline="") as f_docx, open(
            rest_out, "w", encoding="utf-8", newline=""
        ) as f_rest:
            w_docx = csv.DictWriter(f_docx, fieldnames=reader.fieldnames)
            w_rest = csv.DictWriter(f_rest, fieldnames=reader.fieldnames)
            w_docx.writeheader()
            w_rest.writeheader()

            for row in reader:
                total += 1
                val = row.get(col) or ""
                is_docx = str(val).strip().lower().endswith(".docx")
                if is_docx:
                    w_docx.writerow(row)
                    docx_count += 1
                else:
                    w_rest.writerow(row)
                    rest_count += 1

    # Summary
    print(
        f"Done. Input: {in_path}\n"
        f"  -> .docx rows: {docx_count} -> {docx_out}\n"
        f"  -> Rest rows: {rest_count} -> {rest_out}\n"
        f"Total rows processed (excluding header): {total}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
