"""审计 T1 adapter 的 EOS 学习和自由生成行为，不执行训练。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import yaml
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.train_rlvr import configure_qwen_chat_termination  # noqa: E402


def final_supervised_eos_position(row: dict, eos_token_id: int) -> int:
    positions = [
        index
        for index, (token_id, label) in enumerate(zip(row["input_ids"], row["labels"]))
        if token_id == eos_token_id and label == eos_token_id
    ]
    if not positions:
        raise ValueError(f"row {row['id']} has no supervised assistant EOS")
    return positions[-1]


def find_last_subsequence(values: list[int], pattern: list[int]) -> int | None:
    for start in range(len(values) - len(pattern), -1, -1):
        if values[start : start + len(pattern)] == pattern:
            return start
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--adapter",
        type=Path,
        default=PROJECT_DIR / "outputs" / "distill" / "T1" / "final_adapter",
    )
    parser.add_argument("--no-adapter", action="store_true")
    parser.add_argument("--split", choices=["t1_train", "t1_validation"], default="t1_train")
    parser.add_argument("--row-id")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "reports" / "distill" / "t1_generation_audit_local.json",
    )
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("T1 generation audit requires CUDA")

    config = yaml.safe_load((PROJECT_DIR / "configs" / "rlvr.yaml").read_text(encoding="utf-8"))
    cache = load_from_disk(str(PROJECT_DIR / config["t1"]["cache"]))
    split = cache[args.split]
    if args.row_id:
        matches = [row for row in split if row["id"] == args.row_id]
        if not matches:
            raise ValueError(f"row {args.row_id} is not in {args.split}")
        row = matches[0]
    else:
        row = split[0]

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side="left")
    generation_tokens = configure_qwen_chat_termination(tokenizer)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=compute_dtype,
        low_cpu_mem_usage=True,
    )
    model = base if args.no_adapter else PeftModel.from_pretrained(base, str(args.adapter), is_trainable=False)
    model.eval()
    input_device = model.get_input_embeddings().weight.device

    eos_token_id = int(tokenizer.eos_token_id)
    eos_position = final_supervised_eos_position(row, eos_token_id)
    close_think_ids = tokenizer.encode("</think>", add_special_tokens=False)
    close_think_start = find_last_subsequence(row["input_ids"][:eos_position], close_think_ids)
    full_ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=input_device)
    full_mask = torch.tensor([row["attention_mask"]], dtype=torch.long, device=input_device)
    with torch.inference_mode():
        outputs = model(input_ids=full_ids, attention_mask=full_mask, use_cache=False)
        eos_logits = outputs.logits[0, eos_position - 1].float()
        eos_logit = eos_logits[eos_token_id]
        eos_log_probability = eos_logit - torch.logsumexp(eos_logits, dim=-1)
        eos_rank = int((eos_logits > eos_logit).sum().item()) + 1
        diagnostic_positions = list(range(prompt_length := int(row["prompt_length"]), min(prompt_length + 8, len(row["input_ids"]))))
        if close_think_start is not None:
            diagnostic_positions.extend(range(close_think_start, close_think_start + len(close_think_ids)))
        diagnostic_positions.append(eos_position)
        structure_targets = []
        for position in sorted(set(diagnostic_positions)):
            target_id = int(row["input_ids"][position])
            target_logits = outputs.logits[0, position - 1].float()
            target_logit = target_logits[target_id]
            target_log_probability = target_logit - torch.logsumexp(target_logits, dim=-1)
            structure_targets.append(
                {
                    "position": position,
                    "relative_position": position - prompt_length,
                    "token_id": target_id,
                    "token": tokenizer.decode([target_id]).encode("unicode_escape").decode("ascii"),
                    "probability": math.exp(float(target_log_probability.item())),
                    "rank": int((target_logits > target_logit).sum().item()) + 1,
                }
            )
    del outputs, eos_logits, full_ids, full_mask
    torch.cuda.empty_cache()

    prompt_ids = torch.tensor([row["input_ids"][:prompt_length]], dtype=torch.long, device=input_device)
    prompt_mask = torch.ones_like(prompt_ids)
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "eos_token_id": eos_token_id,
        "pad_token_id": int(tokenizer.pad_token_id),
    }
    if args.do_sample:
        generation_kwargs.update(temperature=args.temperature, top_p=args.top_p)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            **generation_kwargs,
        )
    generated_ids = output_ids[0, prompt_length:].tolist()
    terminated = eos_token_id in generated_ids
    if terminated:
        generated_ids = generated_ids[: generated_ids.index(eos_token_id) + 1]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=False)

    audit = {
        "schema_version": "t1_generation_audit/v1",
        "model": args.model,
        "adapter": None if args.no_adapter else str(args.adapter),
        "split": args.split,
        "row_id": row["id"],
        "teacher_completion_tokens": len(row["input_ids"]) - prompt_length,
        "teacher_forced_eos_probability": math.exp(float(eos_log_probability.item())),
        "teacher_forced_eos_rank": eos_rank,
        "structure_targets": structure_targets,
        "generated_tokens": len(generated_ids),
        "terminated_with_im_end": terminated,
        "do_sample": args.do_sample,
        "seed": args.seed,
        "generation_tokens": generation_tokens,
        "decoded": decoded,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in audit.items() if key != "decoded"}, ensure_ascii=False, indent=2))
    print("decoded preview:", json.dumps(decoded[:1000], ensure_ascii=True))
    print("audit:", args.output)


if __name__ == "__main__":
    main()
