#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple


REPO_DIR_RE = re.compile(r"^\d+_.+$")
ANSWER_DIR_RE = re.compile(r"^answer_(\d+)$")
ANSWERING_RE = re.compile(r"<answering>(.*?)</answering>", re.IGNORECASE | re.DOTALL)
FENCED_CODE_RE = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)

KNOWN_SUFFIXES = ("paper-full", "paper-method", "nopaper")


def iter_repo_dirs(root: Path) -> Iterable[Path]:
    for p in sorted(root.iterdir()):
        if p.is_dir() and REPO_DIR_RE.match(p.name):
            yield p


def normalize_name_part(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", raw)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "unknown"


def parse_model_and_suffix(stem: str) -> Tuple[str, str]:
    for suffix in KNOWN_SUFFIXES:
        tail = f"-{suffix}"
        if stem.endswith(tail):
            model = stem[: -len(tail)]
            return model, suffix

    # Fallback: split by last '-', if no '-', treat all as model.
    if "-" in stem:
        model, suffix = stem.rsplit("-", 1)
        return model, suffix
    return stem, "unknown"


def extract_answering_content(answer_text: str) -> Tuple[str, bool]:
    """
    Return (content, used_answering_tag).
    If <answering> is absent, returns full raw answer_text.
    """
    m = ANSWERING_RE.search(answer_text)
    if not m:
        return answer_text, False
    return m.group(1).strip(), True


def extract_code_from_content(content: str) -> str:
    """
    If markdown fenced blocks exist, extract and merge all fenced code.
    Otherwise return content as-is.
    """
    blocks = [b.strip("\n") for b in FENCED_CODE_RE.findall(content)]
    if blocks:
        return "\n\n".join(blocks).strip() + "\n"
    return content.rstrip() + "\n"


def build_output_path(repo_dir: Path, question_id: int, model: str, suffix: str) -> Path:
    out_dir = repo_dir / f"unittest{question_id}" / "llm_implementations"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_part = normalize_name_part(model)
    suffix_part = normalize_name_part(suffix)
    return out_dir / f"llm_{model_part}_{suffix_part}.py"


def process_answer_file(repo_dir: Path, question_id: int, answer_file: Path) -> Tuple[bool, str]:
    raw_text = answer_file.read_text(encoding="utf-8")
    stem = answer_file.stem
    model, suffix = parse_model_and_suffix(stem)

    content, used_answering = extract_answering_content(raw_text)
    code_text = extract_code_from_content(content)

    out_path = build_output_path(repo_dir, question_id, model, suffix)
    out_path.write_text(code_text, encoding="utf-8")

    status = "answering" if used_answering else "fulltxt"
    return used_answering, f"[ok] {answer_file} -> {out_path} ({status})"


def parse_repo_ids_spec(repo_ids_raw: str) -> set[int]:
    selected: set[int] = set()
    for token in (x.strip() for x in repo_ids_raw.split(",")):
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            if not a.isdigit() or not b.isdigit():
                raise ValueError(f"Invalid repo id range token: {token}")
            start, end = int(a), int(b)
            if start > end:
                raise ValueError(f"Invalid repo id range token: {token}")
            selected.update(range(start, end + 1))
        else:
            if not token.isdigit():
                raise ValueError(f"Invalid repo id token: {token}")
            selected.add(int(token))
    if not selected:
        raise ValueError("No valid repo ids found in --repo-ids")
    return selected


def parse_repo_id(repo_dir_name: str) -> Optional[int]:
    m = re.match(r"^(\d+)_", repo_dir_name)
    if not m:
        return None
    return int(m.group(1))


def main():
    parser = argparse.ArgumentParser(
        description="Extract code from answers/*.txt into unittest*/llm_implementations."
    )
    parser.add_argument("--root", type=Path, default="./GeoCodeBench", help="Workspace root.")
    parser.add_argument(
        "--repo-ids",
        type=str,
        default="",
        help="Only process selected repo ids, e.g. '14,18,20-25'. Empty means all repos.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    selected_repo_ids: Optional[set[int]] = None
    if args.repo_ids.strip():
        selected_repo_ids = parse_repo_ids_spec(args.repo_ids)

    total_txt = 0
    used_answering_count = 0
    used_fulltxt_count = 0

    for repo_dir in iter_repo_dirs(root):
        repo_id = parse_repo_id(repo_dir.name)
        if repo_id is None:
            continue
        if selected_repo_ids is not None and repo_id not in selected_repo_ids:
            continue

        answers_root = repo_dir / "answers"
        if not answers_root.exists():
            continue

        for answer_subdir in sorted(answers_root.iterdir()):
            if not answer_subdir.is_dir():
                continue
            m = ANSWER_DIR_RE.match(answer_subdir.name)
            if not m:
                continue
            question_id = int(m.group(1))

            for answer_file in sorted(answer_subdir.glob("*.txt")):
                total_txt += 1
                used_answering, message = process_answer_file(repo_dir, question_id, answer_file)
                if used_answering:
                    used_answering_count += 1
                else:
                    used_fulltxt_count += 1
                print(message)

    print(
        f"[done] total={total_txt} answering={used_answering_count} "
        f"fulltxt_fallback={used_fulltxt_count}"
    )
    return 0


if __name__ == "__main__":
    main()
