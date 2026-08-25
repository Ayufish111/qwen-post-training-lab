"""重载已有 T1_v2 adapter，独立执行两条自由生成结构门禁。"""

from __future__ import annotations

import argparse
import json
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

from src.train_distill import (  # noqa: E402
    configure_qwen_training_tokens,
    find_audit_rows,
    run_free_generation_gate,
)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "rlvr.yaml")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("T1_v2 generation gate requires CUDA")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    t1 = config["t1"]
    gate_config = t1["generation_gate"]
    max_new_tokens = args.max_new_tokens or int(gate_config["max_new_tokens"])
    if max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")

    adapter = resolve_path(args.adapter)
    output = resolve_path(args.output)
    cache = load_from_disk(str(resolve_path(t1["cache"])))
    rows = find_audit_rows(cache, list(gate_config["audit_rows"]))

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
    )
    tokenizer.padding_side = "right"
    generation_tokens = configure_qwen_training_tokens(tokenizer)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=t1["qlora"]["quant_type"],
        bnb_4bit_use_double_quant=bool(t1["qlora"]["double_quant"]),
        bnb_4bit_compute_dtype=compute_dtype,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=compute_dtype,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
    gate = run_free_generation_gate(model, tokenizer, rows, max_new_tokens)
    gate.update(
        {
            "adapter": str(adapter),
            "audit_rows": list(gate_config["audit_rows"]),
            "generation_tokens": generation_tokens,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(gate, ensure_ascii=True, indent=2))
    print("gate:", output)
    if not gate["passed"]:
        raise RuntimeError("T1_v2 free-generation gate failed; RLVR remains blocked")


if __name__ == "__main__":
    main()
