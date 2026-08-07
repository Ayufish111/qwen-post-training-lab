"""Evaluate a Base model or PEFT adapter on Multi-IF Chinese."""

import argparse
import csv
import importlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml

from evaluate_instruction import generate_answer, load_model, sha256_file


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load(
    (PROJECT_DIR / "configs" / "project.yaml").read_text(encoding="utf-8")
)
DEFAULT_DATA = PROJECT_DIR / "data" / "eval" / "multi_if_zh.csv"
DEFAULT_OFFICIAL_REPO = PROJECT_DIR / "third_party" / "Multi-IF"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "reports" / "eval"


def parse_json(value):
    if not isinstance(value, str):
        return value
    return json.loads(value)


def parse_message(value):
    if value in (None, "", "None"):
        return None
    return parse_json(value)


def parse_constraints(row, turn):
    instruction_ids = parse_json(row[f"turn_{turn}_instruction_id_list"])
    kwargs_values = parse_json(row[f"turn_{turn}_kwargs"])
    kwargs = [parse_json(value) for value in kwargs_values]
    if len(instruction_ids) != len(kwargs):
        raise ValueError(
            f"Turn {turn} has {len(instruction_ids)} instruction ids but "
            f"{len(kwargs)} kwargs entries"
        )
    return instruction_ids, kwargs


def load_official_ifeval(repo_path):
    if not (repo_path / "ifeval.py").exists():
        raise FileNotFoundError(
            f"Official Multi-IF code not found at {repo_path}. Clone "
            "https://github.com/facebookresearch/Multi-IF there first."
        )
    sys.path.insert(0, str(repo_path.resolve()))
    ifeval = importlib.import_module("ifeval")
    local_nltk_data = PROJECT_DIR / "nltk_data"
    if local_nltk_data.exists():
        nltk_path = str(local_nltk_data.resolve())
        if nltk_path not in ifeval.nltk.data.path:
            ifeval.nltk.data.path.insert(0, nltk_path)
    return ifeval


def official_repo_revision(repo_path):
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def response_variants(response):
    lines = response.split("\n")
    without_first = "\n".join(lines[1:]).strip()
    without_last = "\n".join(lines[:-1]).strip()
    without_both = "\n".join(lines[1:-1]).strip()
    variants = [response, without_first, without_last, without_both]
    return variants + [variant.replace("*", "") for variant in variants]


def check_constraints(ifeval, response, instruction_ids, kwargs, loose):
    candidates = response_variants(response) if loose else [response]
    results = []
    for index, instruction_id in enumerate(instruction_ids):
        instruction_class = ifeval.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_class(instruction_id)
        instruction.build_description(**kwargs[index])
        passed = any(
            candidate.strip() and instruction.check_following(candidate)
            for candidate in candidates
        )
        results.append(bool(passed))
    return results


def calculate_metrics(results):
    per_turn = defaultdict(
        lambda: {
            "prompts": 0,
            "strict_prompt_correct": 0,
            "strict_instruction_correct": 0,
            "loose_prompt_correct": 0,
            "loose_instruction_correct": 0,
            "instructions": 0,
        }
    )
    for result in results:
        for turn in result["turns"]:
            metrics = per_turn[str(turn["turn"])]
            strict = turn["strict_results"]
            loose = turn["loose_results"]
            metrics["prompts"] += 1
            metrics["instructions"] += len(strict)
            metrics["strict_prompt_correct"] += int(all(strict))
            metrics["strict_instruction_correct"] += sum(strict)
            metrics["loose_prompt_correct"] += int(all(loose))
            metrics["loose_instruction_correct"] += sum(loose)

    summary = {}
    for turn, counts in sorted(per_turn.items(), key=lambda item: int(item[0])):
        prompt_count = counts["prompts"]
        instruction_count = counts["instructions"]
        strict_prompt = counts["strict_prompt_correct"] / prompt_count
        strict_instruction = (
            counts["strict_instruction_correct"] / instruction_count
        )
        loose_prompt = counts["loose_prompt_correct"] / prompt_count
        loose_instruction = (
            counts["loose_instruction_correct"] / instruction_count
        )
        summary[f"turn_{turn}"] = {
            "prompts": prompt_count,
            "instructions": instruction_count,
            "strict_prompt_accuracy": strict_prompt,
            "strict_instruction_accuracy": strict_instruction,
            "loose_prompt_accuracy": loose_prompt,
            "loose_instruction_accuracy": loose_instruction,
            "official_overall_average": (
                strict_prompt
                + strict_instruction
                + loose_prompt
                + loose_instruction
            )
            / 4,
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--model", default=CONFIG["model"]["name"])
    parser.add_argument("--revision", default=CONFIG["model"]["revision"])
    parser.add_argument("--adapter")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--official-repo", type=Path, default=DEFAULT_OFFICIAL_REPO
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=CONFIG["evaluation"]["multi_if_max_new_tokens"],
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing answer file and skip completed row ids.",
    )
    args = parser.parse_args()

    ifeval = load_official_ifeval(args.official_repo)
    with args.data.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("Multi-IF Chinese data is empty")

    tokenizer, model, compute_dtype, load_seconds = load_model(
        args.model, args.revision, args.adapter
    )
    do_sample = CONFIG["evaluation"]["do_sample"]
    max_new_tokens = args.max_new_tokens
    args.output_dir.mkdir(parents=True, exist_ok=True)
    answer_path = args.output_dir / f"{args.experiment_id}_multi_if_zh.jsonl"

    results = []
    completed_ids = set()
    if args.resume and answer_path.exists():
        with answer_path.open("r", encoding="utf-8") as existing_file:
            for line_number, line in enumerate(existing_file, start=1):
                if not line.strip():
                    continue
                try:
                    result = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Skipping incomplete cached line {line_number}")
                    continue
                results.append(result)
                completed_ids.add(result["id"])
        print(f"Resume enabled: loaded {len(completed_ids)} completed rows")

    write_mode = "a" if args.resume else "w"
    with answer_path.open(write_mode, encoding="utf-8") as output_file:
        for row_index, row in enumerate(rows, start=1):
            if row["key"] in completed_ids:
                print(f"[{row_index}/{len(rows)}] {row['key']} skipped")
                continue

            messages = []
            turn_results = []
            for turn in (1, 2, 3):
                user_message = parse_message(row.get(f"turn_{turn}_prompt"))
                if user_message is None:
                    continue
                messages.append(user_message)
                response, generation = generate_answer(
                    tokenizer,
                    model,
                    messages,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                )
                instruction_ids, kwargs = parse_constraints(row, turn)
                strict = check_constraints(
                    ifeval, response, instruction_ids, kwargs, loose=False
                )
                loose = check_constraints(
                    ifeval, response, instruction_ids, kwargs, loose=True
                )
                turn_results.append(
                    {
                        "turn": turn,
                        "prompt": user_message["content"],
                        "response": response,
                        "instruction_ids": instruction_ids,
                        "strict_results": strict,
                        "loose_results": loose,
                        **generation,
                    }
                )
                messages.append({"role": "assistant", "content": response})

            result = {
                "id": row["key"],
                "language": row["language"],
                "turns": turn_results,
            }
            results.append(result)
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_file.flush()
            print(f"[{row_index}/{len(rows)}] {row['key']}")

    summary = {
        "suite": "Multi-IF Chinese",
        "official_dataset": "facebook/Multi-IF",
        "official_rule_code": str(args.official_repo.resolve()),
        "official_rule_revision": official_repo_revision(args.official_repo),
        "experiment_id": args.experiment_id,
        "model": args.model,
        "model_revision": args.revision,
        "adapter": args.adapter,
        "rows": len(results),
        "limit": args.limit,
        "data_sha256": sha256_file(args.data),
        "decoding": {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "enable_thinking": False,
            "compute_dtype": str(compute_dtype).replace("torch.", ""),
        },
        "model_load_seconds": load_seconds,
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "metrics": calculate_metrics(results),
    }
    summary_path = args.output_dir / f"{args.experiment_id}_multi_if_zh_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("answers:", answer_path)
    print("summary:", summary_path)


if __name__ == "__main__":
    main()
