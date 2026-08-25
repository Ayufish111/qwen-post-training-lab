"""把教师 Thinking 轨迹转换成可审计的 T1 训练缓存。

输入数据已经包含 Qwen3 格式的 assistant 消息，其中 `reasoning_content` 是思考过程，
`content` 是最终回答。本文件只负责检查、套用聊天模板、tokenize 和划分训练/验证集，
不会调用第二个模型改写思考过程。
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
RLVR_CONFIG_PATH = PROJECT_DIR / "configs" / "rlvr.yaml"
RLVR_CONFIG = yaml.safe_load(RLVR_CONFIG_PATH.read_text(encoding="utf-8"))

DEFAULT_INPUT = PROJECT_DIR / RLVR_CONFIG["t1"]["input"]
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "cache" / "t1_qwen3_4b_1024"
DEFAULT_MODEL = RLVR_CONFIG["model"]["primary"]
DEFAULT_MAX_LENGTH = 1024
DEFAULT_SEED = int(RLVR_CONFIG.get("seed", 42))


class DistillDataError(ValueError):
    """教师数据虽然被标为 accepted，但仍违反 T1 格式时抛出。"""


def has_severe_repetition_loop(text: str, min_unit_chars: int = 12) -> bool:
    """检测连续三次完全相同、且不是短碎片的思考单元。"""
    units = re.split(r"(?:\r?\n)+|(?<=[。！？!?])", text)
    normalized = [re.sub(r"\s+", "", unit).lower() for unit in units]
    previous = ""
    run = 0
    for unit in normalized:
        if len(unit) < min_unit_chars:
            previous = ""
            run = 0
            continue
        if unit == previous:
            run += 1
        else:
            previous = unit
            run = 1
        if run >= 3:
            return True
    return False


def validate_teacher_audit(row: dict[str, Any]) -> None:
    """再次执行 accepted 数据承诺过的客观质量门禁。"""
    row_id = str(row.get("id") or "")
    teacher = row.get("teacher")
    if not isinstance(teacher, dict):
        raise DistillDataError(f"missing teacher audit for {row_id}")
    if teacher.get("truncated"):
        raise DistillDataError(f"truncated teacher output for {row_id}")
    strict_results = teacher.get("strict_results")
    if not isinstance(strict_results, list) or not strict_results or not all(
        result is True for result in strict_results
    ):
        raise DistillDataError(f"official constraint check failed for {row_id}")
    raw = teacher.get("raw_continuation")
    if not isinstance(raw, str) or "</think>" not in raw:
        raise DistillDataError(f"missing complete reasoning delimiter for {row_id}")
    reasoning = teacher.get("reasoning_content")
    if isinstance(reasoning, str) and has_severe_repetition_loop(reasoning):
        raise DistillDataError(f"severe reasoning repetition loop for {row_id}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _to_list(value: Any) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    return [int(item) for item in value]


def _tokenize_messages(tokenizer: Any, messages: list[dict[str, Any]]) -> dict[str, Any]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=True,
        return_dict=True,
    )
    return {
        "input_ids": _to_list(encoded["input_ids"]),
        "attention_mask": _to_list(encoded["attention_mask"]),
    }


def tokenize_row(
    row: dict[str, Any], tokenizer: Any, max_length: int = DEFAULT_MAX_LENGTH
) -> dict[str, Any]:
    """序列化一条数据，并把所有 prompt token 的 label 设为 -100。"""
    row_id = str(row.get("id") or "")
    validate_teacher_audit(row)
    messages = row.get("messages")
    if not row_id or not isinstance(messages, list) or len(messages) < 2:
        raise DistillDataError(f"invalid messages/id for row {row_id!r}")
    final_message = messages[-1]
    if final_message.get("role") != "assistant":
        raise DistillDataError(f"last message is not assistant for {row_id}")
    reasoning = final_message.get("reasoning_content")
    answer = final_message.get("content")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise DistillDataError(f"empty reasoning_content for {row_id}")
    if not isinstance(answer, str) or not answer.strip():
        raise DistillDataError(f"empty final content for {row_id}")
    teacher = row["teacher"]
    if teacher.get("reasoning_content") != reasoning:
        raise DistillDataError(f"teacher/message reasoning mismatch for {row_id}")
    if teacher.get("content") != answer:
        raise DistillDataError(f"teacher/message answer mismatch for {row_id}")

    full = _tokenize_messages(tokenizer, messages)
    prompt = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        return_dict=True,
    )
    prompt_ids = _to_list(prompt["input_ids"])
    input_ids = full["input_ids"]
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise DistillDataError(
            f"prompt tokens are not a prefix of full dialogue for {row_id}"
        )
    if len(input_ids) > max_length:
        raise DistillDataError(
            f"sequence length {len(input_ids)} exceeds max_length={max_length} for {row_id}"
        )

    return {
        "id": row_id,
        "input_ids": input_ids,
        "attention_mask": full["attention_mask"],
        "labels": [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :],
        "sequence_length": len(input_ids),
        "prompt_length": len(prompt_ids),
    }


def split_tokenized_rows(
    rows: list[dict[str, Any]], seed: int = DEFAULT_SEED, validation_ratio: float = 0.05
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows:
        raise ValueError("no tokenized rows remain after objective filters")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")
    indexes = list(range(len(rows)))
    random.Random(seed).shuffle(indexes)
    validation_size = max(1, round(len(rows) * validation_ratio))
    validation = [rows[index] for index in indexes[:validation_size]]
    train = [rows[index] for index in indexes[validation_size:]]
    return train, validation


def build_cache(
    input_path: Path,
    output_path: Path,
    tokenizer: Any,
    max_length: int = DEFAULT_MAX_LENGTH,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    from datasets import Dataset, DatasetDict

    rows = read_jsonl(input_path)
    tokenized: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for row in rows:
        try:
            tokenized.append(tokenize_row(row, tokenizer, max_length=max_length))
        except (DistillDataError, KeyError, TypeError, ValueError) as exc:
            rejected.append({"id": str(row.get("id") or ""), "reason": str(exc)})

    train, validation = split_tokenized_rows(tokenized, seed=seed)
    dataset = DatasetDict(
        {
            "t1_train": Dataset.from_list(train),
            "t1_validation": Dataset.from_list(validation),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_path))
    audit = {
        "input": str(input_path),
        "input_rows": len(rows),
        "accepted_rows": len(tokenized),
        "rejected_rows": len(rejected),
        "rejected": rejected,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "max_length": max_length,
        "seed": seed,
        "output": str(output_path),
    }
    (output_path.parent / f"{output_path.name}_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(
            f"Output exists: {args.output}. Move it aside before rebuilding; "
            "this command never overwrites an existing cache."
        )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=False,
        local_files_only=args.local_files_only,
    )
    audit = build_cache(
        args.input,
        args.output,
        tokenizer,
        max_length=args.max_length,
        seed=args.seed,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
