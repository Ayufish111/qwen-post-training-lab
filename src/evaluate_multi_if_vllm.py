"""Final Multi-IF evaluation through offline vLLM with a PEFT adapter."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml

from evaluate_multi_if import (
    calculate_metrics,
    check_constraints,
    load_official_ifeval,
    official_repo_revision,
    parse_message,
    parse_constraints,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((PROJECT_DIR / "configs" / "project.yaml").read_text(encoding="utf-8"))
DEFAULT_DATA = PROJECT_DIR / "data" / "eval" / "multi_if_zh.csv"
DEFAULT_OFFICIAL_REPO = PROJECT_DIR / "third_party" / "Multi-IF"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "reports" / "eval_rlvr"


def sha256_file(path):
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_final_content(text, enable_thinking=False):
    """Keep final answer content and discard Qwen3 reasoning text."""
    text = text.replace("<|im_end|>", "").strip()
    if enable_thinking and not thinking_structure_ok(text, enable_thinking=True):
        return ""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].replace("<think>", "").strip()
    if enable_thinking or "<think>" in text:
        return ""
    return text.replace("<think>", "").strip()


def thinking_structure_ok(text, enable_thinking=False):
    """Audit completion markers, accounting for the prompt-supplied opening tag."""
    opens = text.count("<think>")
    closes = text.count("</think>")
    if enable_thinking:
        return closes >= 1 and text.rfind("</think>") > text.rfind("<think>")
    if opens == 0 and closes == 0:
        return True
    return opens == closes and text.rfind("<think>") < text.rfind("</think>")


def load_vllm(model_path, adapter_path, tensor_parallel_size, gpu_memory_utilization):
    try:
        vllm = importlib.import_module("vllm")
    except ImportError as exc:
        raise RuntimeError(
            "vLLM is not installed in this environment; run this evaluator on the "
            "validated AutoDL/Linux vLLM environment."
        ) from exc

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(f"Unable to load local tokenizer from {model_path}: {exc}") from exc

    llm_kwargs = {
        "model": model_path,
        "enable_lora": bool(adapter_path),
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": False,
    }
    if adapter_path:
        llm_kwargs["max_lora_rank"] = 64
    llm = vllm.LLM(**llm_kwargs)
    lora_request = None
    if adapter_path:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("evaluation_adapter", 1, str(adapter_path))
    return vllm, tokenizer, llm, lora_request


def generate_one(vllm, tokenizer, llm, lora_request, messages, max_tokens, enable_thinking):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    sampling = vllm.SamplingParams(
        temperature=0.0,
        top_p=1.0,
        skip_special_tokens=False,
        max_tokens=max_tokens,
        stop_token_ids=[tokenizer.eos_token_id],
    )
    started = time.perf_counter()
    generate_kwargs = {"sampling_params": sampling}
    if lora_request is not None:
        generate_kwargs["lora_request"] = lora_request
    outputs = llm.generate([prompt], **generate_kwargs)
    elapsed = time.perf_counter() - started
    completion = outputs[0].outputs[0]
    token_ids = getattr(completion, "token_ids", ()) or ()
    raw_text = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    final_content = extract_final_content(raw_text, enable_thinking=enable_thinking)
    generated_tokens = len(token_ids)
    finish_reason = getattr(completion, "finish_reason", None)
    stopped_by_limit = (
        finish_reason == "length"
        if finish_reason is not None
        else generated_tokens >= max_tokens
    )
    saw_thinking_tokens = "<think>" in raw_text or "</think>" in raw_text
    return final_content, {
        "raw_text": raw_text,
        "generated_tokens": generated_tokens,
        "generation_seconds": elapsed,
        "natural_eos": not stopped_by_limit,
        "clipped": stopped_by_limit,
        "finish_reason": finish_reason,
        "saw_thinking_tokens": saw_thinking_tokens,
        "thinking_structure_ok": thinking_structure_ok(
            raw_text, enable_thinking=enable_thinking
        ),
        "enable_thinking": enable_thinking,
    }


def generate_with_answer_only_fallback(
    generate_fn, vllm, tokenizer, llm, lora_request, messages, max_tokens
):
    """Retry an empty thinking answer once without reasoning, preserving audit fields."""
    response, primary = generate_fn(
        vllm, tokenizer, llm, lora_request, messages, max_tokens, True
    )
    primary = dict(primary)
    primary_empty = not response.strip()
    primary_clipped = bool(primary.get("clipped"))
    primary_unclosed_thinking = bool(primary.get("saw_thinking_tokens")) and (
        primary.get("thinking_structure_ok") is not True
    )
    if not (primary_empty or primary_clipped or primary_unclosed_thinking):
        primary.update(
            {
                "answer_only_fallback_attempted": False,
                "answer_only_fallback_used": False,
                "primary_empty": False,
                "primary_clipped": primary_clipped,
                "primary_unclosed_thinking": False,
            }
        )
        return response, primary

    fallback_response, fallback = generate_fn(
        vllm, tokenizer, llm, lora_request, messages, max_tokens, False
    )
    generation = dict(fallback)
    generation.update(
        {
            "answer_only_fallback_attempted": True,
            "answer_only_fallback_used": bool(fallback_response.strip()),
            "primary_empty": primary_empty,
            "primary_clipped": primary_clipped,
            "primary_unclosed_thinking": primary_unclosed_thinking,
            "primary_thinking_structure_ok": primary.get("thinking_structure_ok"),
            "primary_generated_tokens": int(primary.get("generated_tokens", 0)),
            "primary_generation_seconds": float(primary.get("generation_seconds", 0.0)),
            "primary_natural_eos": bool(primary.get("natural_eos", False)),
        }
    )
    return (fallback_response if fallback_response.strip() else response), generation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--official-repo", type=Path, default=DEFAULT_OFFICIAL_REPO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--answer-only-fallback",
        action="store_true",
        help="Retry invalid thinking output without reasoning; disabled for native capability evaluation.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    ifeval = load_official_ifeval(args.official_repo)
    with args.data.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("Multi-IF data is empty")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    answer_path = args.output_dir / f"{args.experiment_id}_multi_if_test_vllm.jsonl"
    summary_path = args.output_dir / f"{args.experiment_id}_multi_if_test_vllm_summary.json"
    completed = {}
    if args.resume and answer_path.exists():
        with answer_path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    result = json.loads(line)
                    completed[result["id"]] = result

    started = time.perf_counter()
    vllm, tokenizer, llm, lora_request = load_vllm(
        args.model, args.adapter, args.tensor_parallel_size, args.gpu_memory_utilization
    )
    new_results = []
    write_mode = "a" if args.resume else "w"
    with answer_path.open(write_mode, encoding="utf-8") as output_file:
        for index, row in enumerate(rows, start=1):
            if row["key"] in completed:
                continue
            messages = []
            turns = []
            for turn in (1, 2, 3):
                user_message = parse_message(row.get(f"turn_{turn}_prompt"))
                if user_message is None:
                    continue
                messages.append(user_message)
                if args.answer_only_fallback:
                    response, generation = generate_with_answer_only_fallback(
                        generate_one,
                        vllm,
                        tokenizer,
                        llm,
                        lora_request,
                        messages,
                        args.max_new_tokens,
                    )
                else:
                    response, generation = generate_one(
                        vllm,
                        tokenizer,
                        llm,
                        lora_request,
                        messages,
                        args.max_new_tokens,
                        True,
                    )
                    generation = dict(generation)
                    generation.update({
                        "answer_only_fallback_attempted": False,
                        "answer_only_fallback_used": False,
                        "primary_empty": not response.strip(),
                        "primary_clipped": bool(generation.get("clipped")),
                        "primary_unclosed_thinking": bool(generation.get("saw_thinking_tokens"))
                        and generation.get("thinking_structure_ok") is not True,
                    })
                instruction_ids, kwargs = parse_constraints(row, turn)
                strict = check_constraints(ifeval, response, instruction_ids, kwargs, loose=False)
                loose = check_constraints(ifeval, response, instruction_ids, kwargs, loose=True)
                turns.append({
                    "turn": turn,
                    "prompt": user_message["content"],
                    "response": response,
                    "instruction_ids": instruction_ids,
                    "strict_results": strict,
                    "loose_results": loose,
                    **{key: value for key, value in generation.items() if key != "raw_text"},
                })
                messages.append({"role": "assistant", "content": response})
            result = {"id": row["key"], "language": row["language"], "turns": turns}
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_file.flush()
            new_results.append(result)
            print(f"[{index}/{len(rows)}] {row['key']}", flush=True)

    all_results = list(completed.values()) + new_results
    token_counts = [turn["generated_tokens"] for result in all_results for turn in result["turns"]]
    generation_seconds = [turn["generation_seconds"] for result in all_results for turn in result["turns"]]
    summary = {
        "suite": "Multi-IF Chinese",
        "backend": "vllm",
        "experiment_id": args.experiment_id,
        "model": args.model,
        "adapter": args.adapter,
        "deployment_mode": "dynamic_lora" if args.adapter else "merged_model",
        "rows": len(all_results),
        "data_sha256": sha256_file(args.data),
        "data_rows_requested": len(rows),
        "decoding": {
            "enable_thinking": True,
            "reasoning_parser": "qwen3_compatible_text_extraction",
            "temperature": 0.0,
            "top_p": 1.0,
            "skip_special_tokens": False,
            "max_new_tokens": args.max_new_tokens,
            "multi_turn_history": "final_content_only",
            "answer_only_fallback": args.answer_only_fallback,
        },
        "metrics": calculate_metrics(all_results),
        "generation": {
            "total_seconds": sum(generation_seconds),
            "mean_seconds": statistics.mean(generation_seconds) if generation_seconds else 0.0,
            "mean_generated_tokens": statistics.mean(token_counts) if token_counts else 0.0,
            "total_generated_tokens": sum(token_counts),
            "clipped_count": sum(turn["clipped"] for result in all_results for turn in result["turns"]),
            "empty_count": sum(not turn["response"].strip() for result in all_results for turn in result["turns"]),
            "answer_only_fallback_attempt_count": sum(bool(turn.get("answer_only_fallback_attempted")) for result in all_results for turn in result["turns"]),
            "answer_only_fallback_used_count": sum(bool(turn.get("answer_only_fallback_used")) for result in all_results for turn in result["turns"]),
            "primary_clipped_count": sum(bool(turn.get("primary_clipped", turn.get("clipped"))) for result in all_results for turn in result["turns"]),
            "primary_empty_count": sum(bool(turn.get("primary_empty")) for result in all_results for turn in result["turns"]),
            "primary_unclosed_thinking_count": sum(bool(turn.get("primary_unclosed_thinking")) for result in all_results for turn in result["turns"]),
        },
        "official_repo_revision": official_repo_revision(args.official_repo),
        "elapsed_seconds": time.perf_counter() - started,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("answers:", answer_path)
    print("summary:", summary_path)


if __name__ == "__main__":
    main()
