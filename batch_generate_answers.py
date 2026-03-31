from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv  # type: ignore[import-not-found]
from openai import OpenAI  # type: ignore[import-not-found]

from tqdm import tqdm  # type: ignore[import-not-found]


REPO_DIR_RE = re.compile(r"^\d+_.+")
INDEX_RE = re.compile(r"^(code|template)_?(\d+)\.py$")

SUFFIX_TO_PAPER_FILE = {
    "paper-full": "paper-full.json",
    "paper-method": "paper-method.json",
    "nopaper": None,
}

PROMPT_INTRO_WITH_PAPER = (
    "You are an expert in 3D vision. Please read the paper and think about what to fill in the "
    "****EMPTY**** section of the code. Output format: Returns the function according to the template.\n\n"
)
PROMPT_INTRO_NO_PAPER = (
    "You are an expert in 3D vision. Please think about what to fill in the ****EMPTY**** section of "
    "the code. Output format: Returns the function according to the template.\n\n"
)
PROMPT_BODY = """## CODE with EMPTY:
{code}

## TEMPLATE:
{template}

Please fill out the template above. Your response must strictly follow the format below:
<thinking>
[Your reasoning or thinking content here]
</thinking>

<answering>
[Fill in the complete template here, ensuring that:
1.All required imports and components are included,
2.The structure strictly follows the provided template,
3.The result is a complete, ready-to-use implementation]
</answering>
"""


def parse_models(models_raw: str) -> List[str]:
    models = [x.strip() for x in models_raw.split(",") if x.strip()]
    if not models:
        raise ValueError("No models provided. Use --models model1,model2")
    return models


def parse_suffixes(suffixes_raw: str) -> List[str]:
    suffixes = [x.strip() for x in suffixes_raw.split(",") if x.strip()]
    invalid = [x for x in suffixes if x not in SUFFIX_TO_PAPER_FILE]
    if invalid:
        raise ValueError(
            f"Invalid suffix: {invalid}. Allowed: {list(SUFFIX_TO_PAPER_FILE.keys())}"
        )
    if not suffixes:
        raise ValueError("No suffixes provided.")
    return suffixes


def parse_repo_id(repo_dir_name: str) -> Optional[int]:
    m = re.match(r"^(\d+)_", repo_dir_name)
    if not m:
        return None
    return int(m.group(1))


def parse_repo_ids_spec(repo_ids_raw: str) -> set[int]:
    """
    Parse repo id selection like: "1,3,8-12".
    """
    selected: set[int] = set()
    for token in (x.strip() for x in repo_ids_raw.split(",")):
        if not token:
            continue
        if "-" in token:
            parts = token.split("-", 1)
            if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
                raise ValueError(f"Invalid repo id range token: {token}")
            start, end = int(parts[0]), int(parts[1])
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


def iter_repo_dirs(root: Path) -> Iterable[Path]:
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if p.name.startswith("."):
            continue
        if REPO_DIR_RE.match(p.name):
            yield p


def discover_question_indices(questions_dir: Path) -> List[int]:
    code_indices = set()
    template_indices = set()

    for child in questions_dir.iterdir():
        if not child.is_file():
            continue
        m = INDEX_RE.match(child.name)
        if not m:
            continue
        kind, idx_raw = m.group(1), m.group(2)
        idx = int(idx_raw)
        if kind == "code":
            code_indices.add(idx)
        elif kind == "template":
            template_indices.add(idx)

    return sorted(code_indices & template_indices)


def resolve_qa_file(questions_dir: Path, stem: str, idx: int) -> Path:
    candidates = [
        questions_dir / f"{stem}{idx}.py",
        questions_dir / f"{stem}_{idx}.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Cannot find {stem} file for index={idx} in {questions_dir}")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def llm_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_retries: int,
) -> str:
    client = OpenAI(api_key=api_key, base_url=base_url)

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                timeout=300,
            )
            content = response.choices[0].message.content
            if not content:
                raise RuntimeError("Empty content in completion response.")
            return content
        except Exception as e:
            last_error = e
            if attempt == max_retries:
                break
            sleep_s = min(2 ** (attempt - 1), 8)
            time.sleep(sleep_s)

    raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_error}")


def build_prompt(
    paper_text: str,
    code_text: str,
    template_text: str,
) -> str:
    """When paper is empty (nopaper), omit the ## PAPER section entirely."""
    body = PROMPT_BODY.format(code=code_text, template=template_text)
    if paper_text.strip():
        paper_section = f"## PAPER:\n{paper_text.strip()}\n\n"
        return PROMPT_INTRO_WITH_PAPER + paper_section + body
    return PROMPT_INTRO_NO_PAPER + body


def main():
    parser = argparse.ArgumentParser(description="Batch generate LLM answers.")
    parser.add_argument("--root", type=Path, default="./GeoCodeBench", help="Repo root path.")
    parser.add_argument(
        "--dotenv",
        type=Path,
        default=Path(".env"),
        help="Path to .env file that stores API config.",
    )
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma separated model names, e.g. gpt-5, gemini-2.5-pro",
    )
    parser.add_argument(
        "--suffixes",
        type=str,
        default="paper-full,paper-method,nopaper",
        help="Comma separated suffixes from: paper-full,paper-method,nopaper",
    )
    parser.add_argument(
        "--repo-ids",
        type=str,
        default="",
        help="Only process selected repo ids, e.g. '01,02,03-05'. Empty means all repos.",
    )
    parser.add_argument(
        "--save-prompt",
        action="store_true",
        help="Save rendered prompt text to each repo's questions/ directory.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of parallel LLM requests.",
    )
    parser.add_argument("--max-retries", type=int, default=3, help="Retries for API failures.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if target answer file already exists.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    dotenv_path = args.dotenv

    load_dotenv(dotenv_path=dotenv_path, override=False)
    api_key = os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL")
    if not api_key:
        raise RuntimeError("Missing API key in .env. Use LLM_API_KEY or OPENAI_API_KEY")

    models = parse_models(args.models)
    suffixes = parse_suffixes(args.suffixes)
    selected_repo_ids: Optional[set[int]] = None
    if args.repo_ids.strip():
        selected_repo_ids = parse_repo_ids_spec(args.repo_ids)

    repo_dirs: List[Path] = []
    for repo_dir in iter_repo_dirs(root):
        repo_id = parse_repo_id(repo_dir.name)
        if repo_id is None:
            continue
        if selected_repo_ids is not None and repo_id not in selected_repo_ids:
            continue
        repo_dirs.append(repo_dir)

    all_jobs: List[Tuple[Path, int, str, str]] = []
    for repo_dir in repo_dirs:
        questions_dir = repo_dir / "questions"
        if not questions_dir.exists():
            continue
        indices = discover_question_indices(questions_dir)
        if not indices:
            continue
        for idx in indices:
            for model_name in models:
                for suffix in suffixes:
                    all_jobs.append((repo_dir, idx, model_name, suffix))

    total_tasks = 0
    success_tasks = 0
    progress = tqdm(total=len(all_jobs), desc="Generating", unit="task") if tqdm else None

    futures = {}
    max_workers = max(1, args.max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for repo_dir in repo_dirs:
            questions_dir = repo_dir / "questions"
            if not questions_dir.exists():
                continue
            indices = discover_question_indices(questions_dir)
            if not indices:
                continue

            for idx in indices:
                code_path = resolve_qa_file(questions_dir, "code", idx)
                template_path = resolve_qa_file(questions_dir, "template", idx)
                code_text = read_text(code_path)
                template_text = read_text(template_path)

                answer_dir = repo_dir / "answers" / f"answer_{idx}"
                answer_dir.mkdir(parents=True, exist_ok=True)

                prompts_by_suffix: Dict[str, str] = {}
                valid_suffixes: List[str] = []
                for suffix in suffixes:
                    paper_file = SUFFIX_TO_PAPER_FILE[suffix]
                    if paper_file is None:
                        paper_text = ""
                    else:
                        paper_path = questions_dir / paper_file
                        if not paper_path.exists():
                            print(f"[warn] missing paper file, skip all models: {paper_path}")
                            continue
                        paper_text = read_text(paper_path)

                    prompt = build_prompt(
                        paper_text=paper_text,
                        code_text=code_text,
                        template_text=template_text,
                    )
                    prompts_by_suffix[suffix] = prompt
                    valid_suffixes.append(suffix)
                    if args.save_prompt:
                        prompt_file = questions_dir / f"prompt{idx}-{suffix}.txt"
                        prompt_file.write_text(prompt, encoding="utf-8")

                for model_name in models:
                    for suffix in suffixes:
                        total_tasks += 1
                        if "/" in model_name:
                            model_name = model_name.replace("/", "_")
                        out_file = answer_dir / f"{model_name}-{suffix}.txt"
                        if args.skip_existing and out_file.exists():
                            print(f"[skip] exists: {out_file}")
                            success_tasks += 1
                            if progress:
                                progress.update(1)
                            continue

                        if suffix not in valid_suffixes:
                            if progress:
                                progress.update(1)
                            continue

                        prompt = prompts_by_suffix[suffix]
                        future = executor.submit(
                            llm_chat_completion,
                            base_url=base_url,
                            api_key=api_key,
                            model=model_name,
                            prompt=prompt,
                            max_retries=args.max_retries,
                        )
                        futures[future] = (repo_dir.name, idx, model_name, suffix, out_file)

        for future in as_completed(futures):
            repo_name, idx, model_name, suffix, out_file = futures[future]
            print(f"[run] {repo_name} q{idx} {model_name} {suffix}")
            try:
                answer_text = future.result()
                out_file.write_text(answer_text, encoding="utf-8")
                success_tasks += 1
                print(f"[ok] {out_file}")
            except Exception as e:
                print(f"[err] {repo_name} q{idx} {model_name} {suffix}: {e}", file=sys.stderr)
            finally:
                if progress:
                    progress.update(1)

    if progress:
        progress.close()
    print(f"[done] success={success_tasks} total={total_tasks}")
    return 0


if __name__ == "__main__":
    main()
