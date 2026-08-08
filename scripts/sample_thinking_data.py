"""Generate auditable Qwen3 thinking trajectories for the T1 cold-start.

The script deliberately keeps teacher generation separate from T1 training:
the raw assistant continuation is parsed into reasoning_content and final
content, and only the final content is checked against the official
Multi-IF-compatible checker. Existing JSONL files are never overwritten
unless --resume is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_DIR / "data" / "rlvr" / "constraint_train_2000.jsonl"
DEFAULT_RAW = PROJECT_DIR / "data" / "distill" / "t1_thinking_raw.jsonl"
DEFAULT_ACCEPTED = PROJECT_DIR / "data" / "distill" / "t1_thinking_accepted.jsonl"
DEFAULT_REJECTED = PROJECT_DIR / "data" / "distill" / "t1_thinking_rejected.jsonl"
DEFAULT_AUDIT = PROJECT_DIR / "reports" / "distill" / "t1_generation_audit.json"
DEFAULT_OFFICIAL_REPO = PROJECT_DIR / "third_party" / "Multi-IF"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def difficulty_bucket(row: dict[str, Any], *, first_pass: bool | None = None, attempts: int = 0) -> str:
    """Assign the frozen easy/medium/hard bucket from visible train metadata."""

    metadata = row.get("metadata") or {}
    constraints = int(metadata.get("constraint_count", len(row.get("instruction_ids", []))))
    turns = int(metadata.get("context_turns", 1))
    categories = len(row.get("constraint_categories") or [])
    score = constraints + max(0, turns - 1) + max(0, categories - 1)
    if first_pass is False:
        score += 1
    score += min(attempts, 2)
    if score <= 2:
        return "easy"
    if score <= 4:
        return "medium"
    return "hard"


def parse_thinking_continuation(decoded: str) -> tuple[str, str] | None:
    """Return (reasoning_content, content) from a Qwen3 continuation.

    Qwen3 puts the opening think marker in the prompt when
    enable_thinking=True, so generated text normally starts with reasoning and
    contains only the closing marker. The parser accepts both forms to make
    the smoke test diagnose template/version mismatches explicitly.
    """

    text = decoded.strip()
    text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    if "</think>" not in text:
        return None
    reasoning, answer = text.split("</think>", 1)
    reasoning = reasoning.replace("<think>", "", 1).strip()
    answer = answer.strip()
    if not reasoning or not answer:
        return None
    return reasoning, answer


def load_official_ifeval(repo_path: Path):
    if not (repo_path / "ifeval.py").exists():
        raise FileNotFoundError(f"Official checker not found: {repo_path / 'ifeval.py'}")
    sys.path.insert(0, str(repo_path.resolve()))
    ifeval = importlib.import_module("ifeval")
    local_nltk_data = PROJECT_DIR / "nltk_data"
    if local_nltk_data.exists() and str(local_nltk_data.resolve()) not in ifeval.nltk.data.path:
        ifeval.nltk.data.path.insert(0, str(local_nltk_data.resolve()))
    return ifeval


def check_constraints(ifeval, response: str, instruction_ids: list[str], kwargs: list[dict[str, Any]]) -> list[bool]:
    results = []
    for index, instruction_id in enumerate(instruction_ids):
        instruction_class = ifeval.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_class(instruction_id)
        instruction.build_description(**kwargs[index])
        results.append(bool(response.strip() and instruction.check_following(response)))
    return results


def build_model(model_name: str, revision: str, trust_remote_code: bool):
    if not torch.cuda.is_available():
        raise RuntimeError("T1 teacher generation requires an NVIDIA GPU.")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=trust_remote_code,
        quantization_config=quantization,
        device_map="auto",
        dtype=compute_dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tokenizer, model, compute_dtype


def generate_once(tokenizer, model, messages: list[dict[str, Any]], max_new_tokens: int, seed: int, do_sample: bool):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        return_tensors="pt",
    )
    if isinstance(encoded, dict):
        model_inputs = dict(encoded)
    else:
        model_inputs = {"input_ids": encoded}
    input_device = model.get_input_embeddings().weight.device
    model_inputs = {name: value.to(input_device) for name, value in model_inputs.items()}
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs.update(temperature=0.6, top_p=0.95)
    with torch.inference_mode():
        output_ids = model.generate(**model_inputs, **generation_kwargs)
    new_ids = output_ids[0, model_inputs["input_ids"].shape[1] :]
    decoded = tokenizer.decode(new_ids, skip_special_tokens=False)
    ended_with_eos = bool(len(new_ids) and int(new_ids[-1]) == tokenizer.eos_token_id)
    return decoded, len(new_ids), int(model_inputs["input_ids"].shape[1]), ended_with_eos


def load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {row["id"] for row in read_jsonl(path) if isinstance(row, dict) and "id" in row}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="Qwen/Qwen3-4B-Instruct")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_ACCEPTED)
    parser.add_argument("--rejected-output", type=Path, default=DEFAULT_REJECTED)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--official-repo", type=Path, default=DEFAULT_OFFICIAL_REPO)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.max_retries < 0:
        parser.error("--max-retries cannot be negative")
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    for path in (args.raw_output, args.output, args.rejected_output):
        if path.exists() and not args.resume:
            raise FileExistsError(f"Refusing to overwrite existing JSONL: {path}; use --resume")

    rows = read_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No input rows")
    ifeval = load_official_ifeval(args.official_repo)
    tokenizer, model, compute_dtype = build_model(args.teacher, args.revision, False)
    raw_ids = load_existing_ids(args.raw_output)
    accepted_ids = load_existing_ids(args.output)
    rejected_ids = load_existing_ids(args.rejected_output)
    overlap = accepted_ids & rejected_ids
    if overlap:
        raise RuntimeError(f"IDs exist in both accepted and rejected outputs: {sorted(overlap)[:5]}")
    completed = accepted_ids | rejected_ids
    input_ids = {str(row.get("id", "")) for row in rows}
    initial_completed = completed & input_ids
    initial_completed_count = len(initial_completed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.rejected_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    accepted_count = rejected_count = 0
    attempts_total = 0
    truncations = 0
    reasoning_tokens = []
    answer_tokens = []
    started = time.perf_counter()
    with args.raw_output.open("a", encoding="utf-8") as raw_file, args.output.open("a", encoding="utf-8") as accepted_file, args.rejected_output.open("a", encoding="utf-8") as rejected_file:
        for index, row in enumerate(rows, start=1):
            row_id = str(row.get("id", ""))
            if not row_id:
                raise ValueError(f"Input row {index} has no id")
            if row_id in completed:
                print(f"[{index}/{len(rows)}] {row_id} skipped")
                continue
            final = None
            last_candidate = None
            row_attempts = 0
            last_error = None
            for attempt in range(args.max_retries + 1):
                row_attempts += 1
                attempts_total += 1
                raw_text, generated_count, prompt_count, ended_with_eos = generate_once(
                    tokenizer, model, row["messages"], args.max_new_tokens,
                    seed=42 + index * 100 + attempt, do_sample=args.do_sample,
                )
                was_truncated = generated_count >= args.max_new_tokens and not ended_with_eos
                if was_truncated:
                    truncations += 1
                parsed = parse_thinking_continuation(raw_text)
                if parsed is None:
                    last_error = "missing_or_empty_thinking_delimiter"
                    continue
                reasoning, answer = parsed
                instruction_results = check_constraints(
                    ifeval, answer, row["instruction_ids"], row["kwargs"]
                )
                reasoning_count = len(tokenizer.encode(reasoning, add_special_tokens=False))
                answer_count = len(tokenizer.encode(answer, add_special_tokens=False))
                last_candidate = {
                    "id": row_id,
                    "messages": row["messages"],
                    "instruction_ids": row["instruction_ids"],
                    "kwargs": row["kwargs"],
                    "constraint_categories": row.get("constraint_categories", []),
                    "metadata": row.get("metadata", {}),
                    "teacher": {
                        "model": args.teacher,
                        "revision": args.revision,
                        "raw_continuation": raw_text,
                        "reasoning_content": reasoning,
                        "content": answer,
                        "reasoning_tokens": reasoning_count,
                        "answer_tokens": answer_count,
                        "generated_tokens": generated_count,
                        "prompt_tokens": prompt_count,
                        "strict_results": instruction_results,
                        "first_pass": attempt == 0,
                        "attempts": row_attempts,
                        "difficulty": difficulty_bucket(row, first_pass=attempt == 0, attempts=attempt),
                        "truncated": was_truncated,
                    },
                }
                if was_truncated:
                    last_error = "max_new_tokens_reached"
                    continue
                if all(instruction_results):
                    final = last_candidate
                    break
                last_error = "official_constraint_check_failed"
            if final is None:
                if last_candidate is not None and row_id not in raw_ids:
                    append_jsonl(raw_file, last_candidate)
                    raw_ids.add(row_id)
                rejected = {
                    "id": row_id,
                    "instruction_ids": row.get("instruction_ids", []),
                    "metadata": row.get("metadata", {}),
                    "attempts": row_attempts,
                    "reason": last_error,
                }
                append_jsonl(rejected_file, rejected)
                rejected_count += 1
                print(f"[{index}/{len(rows)}] {row_id} rejected: {last_error}")
            else:
                if row_id not in raw_ids:
                    append_jsonl(raw_file, final)
                    raw_ids.add(row_id)
                append_jsonl(accepted_file, final)
                completed.add(row_id)
                accepted_count += 1
                reasoning_tokens.append(final["teacher"]["reasoning_tokens"])
                answer_tokens.append(final["teacher"]["answer_tokens"])
                print(f"[{index}/{len(rows)}] {row_id} accepted ({row_attempts} attempt(s))")

    audit = {
        "schema_version": "t1_teacher_generation/v1",
        "status": "completed" if (accepted_count + rejected_count) == len(rows) - initial_completed_count else "partial",
        "teacher": args.teacher,
        "revision": args.revision,
        "input": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input),
        "requested_rows": len(rows),
        "accepted_rows_new": accepted_count,
        "rejected_rows_new": rejected_count,
        "attempts_total_new": attempts_total,
        "retry_rate_new": (attempts_total - accepted_count - rejected_count) / attempts_total if attempts_total else 0.0,
        "truncations_new": truncations,
        "reasoning_tokens_mean_new": sum(reasoning_tokens) / len(reasoning_tokens) if reasoning_tokens else None,
        "answer_tokens_mean_new": sum(answer_tokens) / len(answer_tokens) if answer_tokens else None,
        "generation_seconds_new": time.perf_counter() - started,
        "settings": {"enable_thinking": True, "max_new_tokens": args.max_new_tokens, "max_retries": args.max_retries, "do_sample": args.do_sample, "compute_dtype": str(compute_dtype).replace("torch.", "")},
        "outputs": {"raw": str(args.raw_output.resolve()), "accepted": str(args.output.resolve()), "rejected": str(args.rejected_output.resolve())},
    }
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
