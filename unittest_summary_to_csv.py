#!/usr/bin/env python3
"""Aggregate average pass rate per implementation from unittest summaries."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan {repo}/unittest*/test_summary.json and compute average pass rate "
            "for each implementation."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default="./GeoCodeBench",
        help="Workspace root directory to scan (default: script directory).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output CSV path (default: <timestamp>_implementation_avg_pass_rate.csv "
            "under root)."
        ),
    )
    return parser.parse_args()


def collect_summary_files(root: Path) -> List[Path]:
    return sorted(root.glob("[0-9]*_*/unittest*/test_summary.json"))


def aggregate_pass_rate(summary_files: List[Path]) -> List[Dict[str, object]]:
    ratios_by_impl = defaultdict(list)
    files_count_by_impl = defaultdict(int)
    for summary_file in summary_files:
        with summary_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        implementations = data.get("implementations", [])
        for item in implementations:
            name = item.get("name")
            test_total = item.get("test_total")
            test_pass = item.get("test_pass")

            if not isinstance(name, str):
                continue
            if not isinstance(test_total, int) or not isinstance(test_pass, int):
                continue

            files_count_by_impl[name] += 1
            if test_total == 0:
                ratios_by_impl[name].append(0.0)
            else:
                ratios_by_impl[name].append(test_pass / test_total)

    rows: List[Dict[str, object]] = []
    for impl in sorted(files_count_by_impl):
        if impl == "llm_correct":
            continue
        ratios = ratios_by_impl.get(impl, [])
        avg = sum(ratios) / len(ratios) if ratios else 0.0
        if "paper_method" in impl:
            paper_type = "method"
        elif "paper_full" in impl:
            paper_type = "full"
        else:
            paper_type = "nopaper"
        rows.append(
            {
                "implementation": impl,
                "paper_type": paper_type,
                "avg_pass_rate": f"{avg:.6f}",
                "avg_pass_rate_percent": f"{avg * 100:.2f}",
                "unittest_count": len(ratios),
                
            }
        )

    # Highest average first, then by name for stable ordering.
    rows.sort(key=lambda x: (-float(x["avg_pass_rate"]), x["implementation"]))
    return rows


def write_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "implementation",
        "paper_type",
        "avg_pass_rate",
        "avg_pass_rate_percent",
        "unittest_count",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Root directory does not exist: {root}")

    summary_files = collect_summary_files(root)
    if not summary_files:
        raise SystemExit(f"No test_summary.json found under: {root}")

    output_path = args.output
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{timestamp}_implementation_avg_pass_rate.csv"
    else:
        output_path = output_path.resolve()

    rows = aggregate_pass_rate(summary_files)
    write_csv(rows, output_path)

    print(f"Scanned summaries: {len(summary_files)}")
    print(f"Implementations: {len(rows)}")
    print(f"Wrote CSV: {output_path}")


if __name__ == "__main__":
    main()
