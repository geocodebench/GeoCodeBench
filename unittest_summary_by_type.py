#!/usr/bin/env python3
"""Aggregate LLM pass rates by implementation across question types."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan {repo}/unittest*/test_summary.json and compute per-implementation "
            "average pass rate for each question type."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Workspace root directory to scan (default: script directory).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output CSV path (default: <timestamp>_implementation_avg_pass_rate_by_"
            "question_type.csv under root)."
        ),
    )
    parser.add_argument(
        "--repos-csv",
        type=Path,
        default=None,
        help=(
            "Path to repos.csv (default: try <root>/GeoCodeBench/repos.csv, "
            "then <root>/repos.csv)."
        ),
    )
    return parser.parse_args()


def collect_summary_files(root: Path) -> List[Path]:
    return sorted(root.glob("[0-9]*_*/unittest*/test_summary.json"))


def resolve_repos_csv(root: Path, repos_csv_arg: Optional[Path]) -> Path:
    if repos_csv_arg is not None:
        return repos_csv_arg.resolve()

    candidates = [root / "GeoCodeBench" / "repos.csv", root / "repos.csv"]
    for path in candidates:
        if path.is_file():
            return path
    raise SystemExit(
        "repos.csv not found. Please set --repos-csv explicitly."
    )


def parse_unittest_index(unittest_dir_name: str) -> Optional[int]:
    match = re.fullmatch(r"unittest(\d*)", unittest_dir_name)
    if not match:
        return None
    suffix = match.group(1)
    return int(suffix) if suffix else 1


def load_repos_metadata(repos_csv_path: Path) -> Dict[Tuple[str, int], Tuple[str, str]]:
    if not repos_csv_path.is_file():
        raise SystemExit(f"repos.csv does not exist: {repos_csv_path}")

    metadata: Dict[Tuple[str, int], Tuple[str, str]] = {}
    with repos_csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            repo_name = (row.get("repo_name") or "").strip()
            index_text = (row.get("unittest_index") or "").strip()
            source = (row.get("Source") or "").strip()
            question_type = (row.get("Question Type") or "").strip()

            if not repo_name or not index_text:
                continue
            try:
                unittest_index = int(index_text)
            except ValueError:
                continue
            metadata[(repo_name, unittest_index)] = (
                source or "Unknown",
                question_type or "Unknown",
            )
    return metadata


def is_llm_implementation(name: str) -> bool:
    return name.startswith("llm_") and name != "llm_correct"


def classify_paper_type(implementation_name: str) -> str:
    if "paper_method" in implementation_name:
        return "method"
    if "paper_full" in implementation_name:
        return "full"
    return "nopaper"


def extract_question_types(repos_metadata: Dict[Tuple[str, int], Tuple[str, str]]) -> List[str]:
    preferred_order = [
        "Geometric Transformations",
        "Mechanics/Optics Formulation",
        "Geometric Logic Routing",
        "Novel Algorithm Implementation",
    ]
    all_types = {question_type for _, question_type in repos_metadata.values()}
    ordered = [q for q in preferred_order if q in all_types]
    remaining = sorted(all_types - set(preferred_order))
    return ordered + remaining


def aggregate_pass_rate_by_impl_question_type(
    summary_files: List[Path], repos_metadata: Dict[Tuple[str, int], Tuple[str, str]]
) -> List[Dict[str, object]]:
    ratios_by_impl_and_type = defaultdict(list)
    for summary_file in summary_files:
        with summary_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        repo_name = summary_file.parent.parent.name
        unittest_dir_name = summary_file.parent.name
        unittest_index = parse_unittest_index(unittest_dir_name)
        question_type = "Unknown"
        if unittest_index is not None:
            _, question_type = repos_metadata.get(
                (repo_name, unittest_index), ("Unknown", "Unknown")
            )

        implementations = data.get("implementations", [])
        for item in implementations:
            name = item.get("name")
            test_total = item.get("test_total")
            test_pass = item.get("test_pass")

            if not isinstance(name, str):
                continue
            if not isinstance(test_total, int) or not isinstance(test_pass, int):
                continue
            if not is_llm_implementation(name):
                continue

            group_key = (name, question_type)
            if test_total == 0:
                ratios_by_impl_and_type[group_key].append(0.0)
            else:
                ratios_by_impl_and_type[group_key].append(test_pass / test_total)

    question_types = extract_question_types(repos_metadata)
    implementations = sorted({impl for impl, _ in ratios_by_impl_and_type.keys()})

    rows: List[Dict[str, object]] = []
    for implementation in implementations:
        row: Dict[str, object] = {
            "implementation": implementation,
            "paper_type": classify_paper_type(implementation),
        }
        overall_ratios: List[float] = []
        for question_type in question_types:
            ratios = ratios_by_impl_and_type.get((implementation, question_type), [])
            if ratios:
                avg = sum(ratios) / len(ratios)
                row[f"Avg Passrate of {question_type}"] = f"{avg:.6f}"
                overall_ratios.extend(ratios)
            else:
                row[f"Avg Passrate of {question_type}"] = ""
        overall_avg = sum(overall_ratios) / len(overall_ratios) if overall_ratios else 0.0
        row["_overall_avg"] = overall_avg
        rows.append(row)

    rows.sort(key=lambda x: (-float(x["_overall_avg"]), str(x["implementation"])))
    return rows, question_types


def write_csv(rows: List[Dict[str, object]], question_types: List[str], output_path: Path) -> None:
    fieldnames = ["implementation", "paper_type"] + [
        f"Avg Passrate of {question_type}" for question_type in question_types
    ]
    csv_rows: List[Dict[str, object]] = []
    for row in rows:
        clean_row = {field: row.get(field, "") for field in fieldnames}
        csv_rows.append(clean_row)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Root directory does not exist: {root}")

    summary_files = collect_summary_files(root)
    if not summary_files:
        raise SystemExit(f"No test_summary.json found under: {root}")

    repos_csv_path = resolve_repos_csv(root, args.repos_csv)
    repos_metadata = load_repos_metadata(repos_csv_path)
    rows, question_types = aggregate_pass_rate_by_impl_question_type(
        summary_files, repos_metadata
    )

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = root / f"{timestamp}_implementation_avg_pass_rate_by_question_type.csv"
    else:
        output_path = args.output.resolve()
    write_csv(rows, question_types, output_path)

    print(f"Scanned summaries: {len(summary_files)}")
    print(f"Implementations: {len(rows)}")
    print(f"Question types: {len(question_types)}")
    print(f"Repos metadata: {repos_csv_path}")
    print(f"Wrote CSV: {output_path}")


if __name__ == "__main__":
    main()
