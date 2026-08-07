import argparse
import hashlib
import json
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROJECT_CONFIG = yaml.safe_load(
    (PROJECT_DIR / "configs" / "project.yaml").read_text(encoding="utf-8")
)
MODEL_NAME = PROJECT_CONFIG["model"]["name"]
MODEL_REVISION = PROJECT_CONFIG["model"]["revision"]
TRUST_REMOTE_CODE = PROJECT_CONFIG["model"]["trust_remote_code"]
EVALUATION_CONFIG = PROJECT_CONFIG["evaluation"]
DEFAULT_DATA_PATH = PROJECT_DIR / "data" / "eval" / "instruction_following.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "reports" / "eval"


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_answer(answer, check):
    check_type = check["type"]
    text = answer.strip()

    if check_type == "exact":
        return text == check["value"]
    if check_type == "contains_all":
        return all(value in text for value in check["values"])
    if check_type == "excludes_all":
        return all(value not in text for value in check["values"])
    if check_type == "line_count":
        lines = [line for line in text.splitlines() if line.strip()]
        return len(lines) == check["value"]
    if check_type == "numbered_list":
        lines = [line for line in text.splitlines() if line.strip()]
        return len(lines) == check["value"] and all(
            re.match(rf"^{index}\.\s", line) for index, line in enumerate(lines, 1)
        )
    if check_type == "bullet_list":
        lines = [line for line in text.splitlines() if line.strip()]
        return len(lines) == check["value"] and all(line.startswith("- ") for line in lines)
    if check_type == "json_value":
        try:
            return json.loads(text) == check["value"]
        except json.JSONDecodeError:
            return False
    if check_type == "char_range":
        length = len(text)
        return check["min"] <= length <= check["max"]
    if check_type == "regex_forbidden":
        return re.search(check["value"], text) is None

    raise ValueError(f"Unknown check type: {check_type}")


def load_model(model_name, revision, adapter_path):
    if not torch.cuda.is_available():
        raise RuntimeError("This evaluator requires an NVIDIA GPU.")

    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=TRUST_REMOTE_CODE,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=TRUST_REMOTE_CODE,
        quantization_config=quantization_config,
        device_map="auto",
        dtype=compute_dtype,
        low_cpu_mem_usage=True,
    )

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    return tokenizer, model, compute_dtype, load_seconds


def generate_answer(tokenizer, model, messages, max_new_tokens, do_sample):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    model_inputs = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_device = model.get_input_embeddings().weight.device
    model_inputs = {name: value.to(input_device) for name, value in model_inputs.items()}

    torch.cuda.synchronize()
    generation_started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - generation_started

    new_ids = output_ids[0, model_inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    metrics = {
        "prompt_tokens": model_inputs["input_ids"].shape[1],
        "generated_tokens": len(new_ids),
        "generation_seconds": generation_seconds,
    }
    return answer, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", default="B0")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--adapter")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=EVALUATION_CONFIG["max_new_tokens"],
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    print("WARNING: smoke_instruction_20 is a regression check, not a formal benchmark.")

    examples = read_jsonl(args.data)
    if args.limit is not None:
        examples = examples[:args.limit]
    if not examples:
        raise ValueError("The evaluation file contains no examples to run.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / f"{args.experiment_id}_answers.jsonl"

    started_at_utc = datetime.now(timezone.utc).isoformat()
    run_started = time.perf_counter()
    tokenizer, model, compute_dtype, load_seconds = load_model(
        args.model,
        args.revision,
        args.adapter,
    )
    results = []
    category_scores = defaultdict(list)
    do_sample = EVALUATION_CONFIG["do_sample"]

    with result_path.open("w", encoding="utf-8") as file:
        for index, example in enumerate(examples, 1):
            answer, generation_metrics = generate_answer(
                tokenizer,
                model,
                example["messages"],
                args.max_new_tokens,
                do_sample,
            )
            check_results = [
                check_answer(answer, check) for check in example["checks"]
            ]
            passed = all(check_results)
            category_scores[example["category"]].append(passed)
            result = {
                "id": example["id"],
                "category": example["category"],
                "answer": answer,
                "passed": passed,
                "check_results": check_results,
                **generation_metrics,
            }
            results.append(result)
            file.write(json.dumps(result, ensure_ascii=False) + "\n")
            file.flush()
            print(
                f"[{index}/{len(examples)}] {example['id']}: "
                f"{'PASS' if passed else 'FAIL'}"
            )

    generation_seconds = sum(
        result["generation_seconds"] for result in results
    )
    generated_tokens = sum(result["generated_tokens"] for result in results)
    total_seconds = time.perf_counter() - run_started

    summary = {
        "evaluation_suite": "smoke_instruction_20",
        "formal_benchmark": False,
        "experiment_id": args.experiment_id,
        "model": args.model,
        "model_revision": args.revision,
        "adapter": args.adapter,
        "run_config": {
            "started_at_utc": started_at_utc,
            "data_path": str(args.data.resolve()),
            "data_sha256": sha256_file(args.data),
            "limit": args.limit,
            "quantization": "4bit-nf4-double-quant",
            "compute_dtype": str(compute_dtype).replace("torch.", ""),
            "max_new_tokens": args.max_new_tokens,
            "do_sample": do_sample,
            "enable_thinking": False,
            "torch_version": str(torch.__version__),
            "transformers_version": transformers.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "total": len(results),
        "passed": sum(result["passed"] for result in results),
        "pass_rate": sum(result["passed"] for result in results) / len(results),
        "category_pass_rate": {
            category: sum(scores) / len(scores)
            for category, scores in sorted(category_scores.items())
        },
        "runtime": {
            "model_load_seconds": load_seconds,
            "generation_seconds": generation_seconds,
            "total_seconds": total_seconds,
            "generated_tokens": generated_tokens,
            "generation_tokens_per_second": (
                generated_tokens / generation_seconds
                if generation_seconds > 0
                else None
            ),
            "peak_gpu_memory_gb": (
                torch.cuda.max_memory_allocated() / 1024**3
            ),
        },
    }
    summary_path = args.output_dir / f"{args.experiment_id}_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("answers:", result_path)
    print("summary:", summary_path)


if __name__ == "__main__":
    main()
