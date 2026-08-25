"""R0/R1/R2 共用的数据契约、实验约束和批量奖励适配器。

本文件故意不依赖 torch、transformers 或 TRL，因此 Windows 本机不用加载 4B 模型也能检查：
三组实验开关是否符合归因要求、JSONL 数据格式是否正确、传给 TRL 的列是否完整，以及奖励函数
能否处理一批生成结果。真正的模型 rollout 和反向传播仍然在 AutoDL 上完成。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from src.rlvr_rewards import RewardCalculator, soft_overlong_penalty


EXPERIMENT_IDS = ("R0", "R1", "R2")


@dataclass(frozen=True)
class AlgorithmContract:
    experiment_id: str
    name: str
    clip_higher: bool
    token_level_loss: bool
    soft_overlong_penalty: bool
    constraint_aware_sampling: bool


@dataclass(frozen=True)
class RewardRecord:
    row_id: str
    completion_index: int
    parse_ok: bool
    reasoning_missing: bool
    answer_only_fallback: bool
    r_instruction: float
    r_prompt: float
    r_core: float
    overlong_penalty: float
    reward: float
    completion_tokens: int
    strict_results: tuple[bool, ...]
    instruction_ids: tuple[str, ...]


def load_algorithm_contract(config: dict[str, Any], experiment_id: str) -> AlgorithmContract:
    """读取一组算法开关，并同时检查 R1/R2 的归因边界。"""
    if experiment_id not in EXPERIMENT_IDS:
        raise ValueError(f"Unknown experiment {experiment_id!r}; expected one of {EXPERIMENT_IDS}")
    algorithms = config["rlvr"]["algorithms"]
    try:
        row = algorithms[experiment_id]
    except KeyError as exc:
        raise ValueError(f"Missing rlvr.algorithms.{experiment_id}") from exc
    required = {
        "name",
        "clip_higher",
        "token_level_loss",
        "soft_overlong_penalty",
        "constraint_aware_sampling",
    }
    missing = required - set(row)
    if missing:
        raise ValueError(f"{experiment_id} is missing algorithm fields: {sorted(missing)}")
    contract = AlgorithmContract(
        experiment_id=experiment_id,
        name=str(row["name"]),
        clip_higher=bool(row["clip_higher"]),
        token_level_loss=bool(row["token_level_loss"]),
        soft_overlong_penalty=bool(row["soft_overlong_penalty"]),
        constraint_aware_sampling=bool(row["constraint_aware_sampling"]),
    )
    validate_algorithm_matrix(config)
    return contract


def validate_algorithm_matrix(config: dict[str, Any]) -> None:
    """强制 R2 与 R1 只能在“是否使用约束感知采样”这一项上不同。"""
    algorithms = config["rlvr"]["algorithms"]
    if set(algorithms) != set(EXPERIMENT_IDS):
        raise ValueError("rlvr.algorithms must contain exactly R0, R1, and R2")
    r0, r1, r2 = (algorithms[key] for key in EXPERIMENT_IDS)
    expected_r0 = {
        "clip_higher": False,
        "token_level_loss": False,
        "soft_overlong_penalty": False,
        "constraint_aware_sampling": False,
    }
    for key, expected in expected_r0.items():
        if bool(r0[key]) is not expected:
            raise ValueError(f"R0.{key} must be {expected}")
    for key in (
        "clip_higher",
        "token_level_loss",
        "soft_overlong_penalty",
    ):
        if not bool(r1[key]) or bool(r1[key]) != bool(r2[key]):
            raise ValueError(f"R1 and R2 must both enable {key}")
    if bool(r1["constraint_aware_sampling"]):
        raise ValueError("R1 must use uniform proposal sampling")
    if not bool(r2["constraint_aware_sampling"]):
        raise ValueError("R2 must be the only constraint-aware sampler")
    comparable_r1 = {key: value for key, value in r1.items() if key not in {"name", "constraint_aware_sampling"}}
    comparable_r2 = {key: value for key, value in r2.items() if key not in {"name", "constraint_aware_sampling"}}
    if comparable_r1 != comparable_r2:
        raise ValueError("R1/R2 may differ only in name and constraint_aware_sampling")


def validate_rlvr_config(config: dict[str, Any]) -> None:
    """导入 TRL 前，检查三组正式实验共同使用的冻结参数。"""
    validate_algorithm_matrix(config)
    rlvr = config["rlvr"]
    if rlvr.get("initialization") not in {"fresh_lora_on_qwen3_4b", "existing_adapter"}:
        raise ValueError("unsupported rlvr.initialization")
    has_initial_adapter = bool(rlvr.get("initial_adapter"))
    if rlvr["initialization"] == "fresh_lora_on_qwen3_4b" and has_initial_adapter:
        raise ValueError("fresh_lora_on_qwen3_4b requires rlvr.initial_adapter=null")
    if rlvr["initialization"] == "existing_adapter" and not has_initial_adapter:
        raise ValueError("existing_adapter requires rlvr.initial_adapter")
    qlora = rlvr.get("qlora")
    if not isinstance(qlora, dict) or qlora.get("target_modules") != "all-linear":
        raise ValueError("rlvr.qlora.target_modules must be all-linear")
    for key in ("r", "lora_alpha"):
        if not isinstance(qlora.get(key), int) or qlora[key] <= 0:
            raise ValueError(f"rlvr.qlora.{key} must be a positive integer")
    if not isinstance(qlora.get("lora_dropout"), (int, float)) or not 0.0 <= float(qlora["lora_dropout"]) < 1.0:
        raise ValueError("rlvr.qlora.lora_dropout must be in [0, 1)")
    positive_ints = (
        "train_steps",
        "completions_per_prompt",
        "maximum_prompt_tokens",
        "maximum_completion_tokens",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "logging_steps",
        "save_steps",
        "save_total_limit",
        "overlong_buffer_tokens",
        "zero_variance_abort_patience",
    )
    for key in positive_ints:
        if not isinstance(rlvr.get(key), int) or rlvr[key] <= 0:
            raise ValueError(f"rlvr.{key} must be a positive integer")
    for key in ("temperature", "top_p", "learning_rate"):
        value = rlvr.get(key)
        if not isinstance(value, (int, float)) or float(value) <= 0.0:
            raise ValueError(f"rlvr.{key} must be positive")
    if float(rlvr["top_p"]) > 1.0:
        raise ValueError("rlvr.top_p must not exceed 1")
    if rlvr["overlong_buffer_tokens"] >= rlvr["maximum_completion_tokens"]:
        raise ValueError("overlong buffer must be shorter than maximum completion length")
    if rlvr["zero_variance_abort_patience"] > 10:
        raise ValueError("zero_variance_abort_patience must not exceed 10 on paid single-GPU runs")
    tolerance = rlvr.get("zero_variance_tolerance")
    if not isinstance(tolerance, (int, float)) or not 0.0 < float(tolerance) < 1.0:
        raise ValueError("rlvr.zero_variance_tolerance must be in (0, 1)")
    if rlvr.get("zero_variance_policy") != "accept_all_batches_abort_on_consecutive_zero_signal":
        raise ValueError("unsupported rlvr.zero_variance_policy")
    effective_prompt_batch = (
        rlvr["per_device_train_batch_size"] * rlvr["gradient_accumulation_steps"]
    )
    if effective_prompt_batch % rlvr["completions_per_prompt"] != 0:
        raise ValueError(
            "effective prompt batch must be divisible by completions_per_prompt "
            "for single-GPU TRL GRPO"
        )
    if rlvr["rollout_backend"] != "hf":
        raise ValueError("The frozen training rollout backend must be hf")
    if rlvr.get("rollout_enable_thinking") is not True:
        raise ValueError("RLVR rollout_enable_thinking must match the T1 thinking template")
    if rlvr["final_evaluation_backend"] != "vllm":
        raise ValueError("The frozen final evaluation backend must be vllm")
    smoke_ids = rlvr.get("smoke_prompt_ids")
    minimum_smoke_ids = 4 if int(config["data"].get("train_size", 4)) >= 4 else 1
    if not isinstance(smoke_ids, list) or len(smoke_ids) < minimum_smoke_ids or len(smoke_ids) != len(set(smoke_ids)):
        raise ValueError(f"rlvr.smoke_prompt_ids must contain at least {minimum_smoke_ids} unique ids")
    if not isinstance(rlvr.get("smoke_selection_policy"), str) or not rlvr["smoke_selection_policy"]:
        raise ValueError("rlvr.smoke_selection_policy must be a non-empty string")
    if int(rlvr["seed"]) != int(config["seed"]):
        raise ValueError("top-level and rlvr seeds must match")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_training_row(row: dict[str, Any]) -> None:
    """检查一条冻结的 RLVR prompt 的结构，此处不调用官方约束检查器。"""
    row_id = row.get("id")
    messages = row.get("messages")
    instruction_ids = row.get("instruction_ids")
    kwargs_list = row.get("kwargs")
    categories = row.get("constraint_categories")
    metadata = row.get("metadata")
    if not isinstance(row_id, str) or not row_id:
        raise ValueError("RLVR row has an invalid id")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{row_id}: messages must be a non-empty list")
    if messages[-1].get("role") != "user":
        raise ValueError(f"{row_id}: final message must be a user rollout prompt")
    if any(
        not isinstance(message, dict)
        or message.get("role") not in {"system", "user", "assistant"}
        or not isinstance(message.get("content"), str)
        or not message["content"].strip()
        for message in messages
    ):
        raise ValueError(f"{row_id}: invalid conversation message")
    if not isinstance(instruction_ids, list) or not instruction_ids:
        raise ValueError(f"{row_id}: instruction_ids must be non-empty")
    if not isinstance(kwargs_list, list) or len(kwargs_list) != len(instruction_ids):
        raise ValueError(f"{row_id}: kwargs must align with instruction_ids")
    if any(not isinstance(value, dict) for value in kwargs_list):
        raise ValueError(f"{row_id}: each constraint kwargs value must be a dict")
    expected_categories = sorted({value.split(":", 1)[0] for value in instruction_ids})
    if categories != expected_categories:
        raise ValueError(
            f"{row_id}: constraint_categories {categories!r} != {expected_categories!r}"
        )
    if not isinstance(metadata, dict):
        raise ValueError(f"{row_id}: metadata must be a dict")
    if metadata.get("constraint_count") != len(instruction_ids):
        raise ValueError(f"{row_id}: constraint_count does not match instruction_ids")
    if metadata.get("context_turns") != sum(
        message["role"] == "user" for message in messages
    ):
        raise ValueError(f"{row_id}: context_turns does not match messages")


def load_training_pool(path: Path, expected_rows: int | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    for row in rows:
        validate_training_row(row)
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate RLVR row ids")
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows} rows, found {len(rows)}")
    return rows


def to_trl_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """转换成 TRL 所需列名，同时保留奖励和采样器需要的全部元数据。"""
    converted = []
    for row in rows:
        validate_training_row(row)
        converted.append(
            {
                "row_id": row["id"],
                "prompt": row["messages"],
                "instruction_ids": row["instruction_ids"],
                "constraint_kwargs": row["kwargs"],
                "constraint_categories": row["constraint_categories"],
                "metadata": row["metadata"],
            }
        )
    return converted


def completion_to_text(completion: Any) -> str:
    """把 TRL 可能返回的字符串、字典或对话列表统一提取成文本。"""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict) and isinstance(completion.get("content"), str):
        reasoning = completion.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning.strip():
            return f"<think>{reasoning}</think>\n{completion['content']}"
        return completion["content"]
    if isinstance(completion, list) and completion:
        final = completion[-1]
        if isinstance(final, dict) and isinstance(final.get("content"), str):
            reasoning = final.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning.strip():
                return f"<think>{reasoning}</think>\n{final['content']}"
            return final["content"]
    raise ValueError(f"Unsupported completion container: {type(completion).__name__}")


class RewardBatchAdapter:
    """连接 TRL 与约束奖励器，并保存每条回答的可审计评分记录。"""

    def __init__(
        self,
        calculator: RewardCalculator,
        contract: AlgorithmContract,
        token_counter: Callable[[str], int],
        maximum_completion_tokens: int,
        overlong_buffer_tokens: int,
    ) -> None:
        self.calculator = calculator
        self.contract = contract
        self.token_counter = token_counter
        self.maximum_completion_tokens = maximum_completion_tokens
        self.overlong_buffer_tokens = overlong_buffer_tokens
        self.records: list[RewardRecord] = []

    def __call__(
        self,
        completions: Sequence[Any],
        instruction_ids: Sequence[Sequence[str]],
        constraint_kwargs: Sequence[Sequence[dict[str, Any]]],
        row_id: Sequence[str],
        **extra: Any,
    ) -> list[float]:
        lengths = {len(completions), len(instruction_ids), len(constraint_kwargs), len(row_id)}
        if len(lengths) != 1:
            raise ValueError("TRL reward columns and completions must have equal lengths")
        rewards: list[float] = []
        completion_ids = extra.get("completion_ids")
        if completion_ids is not None and len(completion_ids) != len(completions):
            raise ValueError("TRL completion_ids and completions must have equal lengths")
        for index, completion in enumerate(completions):
            text = completion_to_text(completion)
            row = {
                "instruction_ids": list(instruction_ids[index]),
                # Hugging Face Dataset 会把异构 kwargs 合并成统一 Arrow struct，
                # 并给不相关字段补 None。恢复原始稀疏字典，否则 checker 会收到
                # end_phrase=None、keyword=None 等无关参数并全部抛 TypeError。
                "kwargs": [
                    {key: value for key, value in item.items() if value is not None}
                    for item in constraint_kwargs[index]
                ],
            }
            score, parsed = self.calculator.reward(row, text)
            reasoning_missing = False
            answer_only_fallback = False
            # TRL 的结构化 Qwen3 响应可能只有 content，没有 reasoning_content。
            # 这种回答仍有可验证的最终答案：给约束 reward，但单独记录 thinking 缺失，
            # 防止整批 reward 归零导致 RL 无梯度。
            answer_only = None
            if isinstance(completion, dict):
                if isinstance(completion.get("content"), str) and not completion.get("reasoning_content"):
                    answer_only = completion["content"]
            elif isinstance(completion, list) and completion:
                final = completion[-1]
                if isinstance(final, dict) and isinstance(final.get("content"), str) and not final.get("reasoning_content"):
                    answer_only = final["content"]
            if parsed is None and answer_only and answer_only.strip():
                strict_results = self.calculator.check_constraints(
                    answer_only, row["instruction_ids"], row["kwargs"]
                )
                score = self.calculator.score(strict_results)
                reasoning_missing = True
            elif parsed is None and isinstance(completion, str) and text.strip():
                # TRL 无 response schema 时会直接传纯字符串；先给最终文本打约束分，
                # 同时记录这是 answer-only fallback，而不是完整 thinking 轨迹。
                strict_results = self.calculator.check_constraints(
                    text, row["instruction_ids"], row["kwargs"]
                )
                score = self.calculator.score(strict_results)
                reasoning_missing = True
                answer_only_fallback = True
            completion_tokens = (
                len(completion_ids[index])
                if completion_ids is not None
                else int(self.token_counter(text))
            )
            length_penalty = 0.0
            if self.contract.soft_overlong_penalty:
                length_penalty = soft_overlong_penalty(
                    completion_tokens,
                    self.maximum_completion_tokens,
                    self.overlong_buffer_tokens,
                )
            reward = score.r_core + length_penalty
            rewards.append(reward)
            self.records.append(
                RewardRecord(
                    row_id=str(row_id[index]),
                    completion_index=index,
                    parse_ok=parsed is not None,
                    reasoning_missing=reasoning_missing,
                    answer_only_fallback=answer_only_fallback,
                    r_instruction=score.r_instruction,
                    r_prompt=score.r_prompt,
                    r_core=score.r_core,
                    overlong_penalty=length_penalty,
                    reward=reward,
                    completion_tokens=completion_tokens,
                    strict_results=score.strict_results,
                    instruction_ids=tuple(str(value) for value in instruction_ids[index]),
                )
            )
        return rewards

    def category_pass_rates(self, records: Sequence[RewardRecord] | None = None) -> dict[str, float]:
        """按顶层约束类别汇总严格通过率，供 R2 更新采样权重。"""
        selected = self.records if records is None else records
        totals: dict[str, list[int]] = {}
        for record in selected:
            for instruction_id, passed in zip(record.instruction_ids, record.strict_results):
                category = instruction_id.split(":", 1)[0]
                bucket = totals.setdefault(category, [0, 0])
                bucket[0] += int(passed)
                bucket[1] += 1
        return {
            category: passed / count
            for category, (passed, count) in sorted(totals.items())
            if count
        }

    def pop_records(self) -> list[RewardRecord]:
        records, self.records = self.records, []
        return records

    @staticmethod
    def serializable(records: Sequence[RewardRecord]) -> list[dict[str, Any]]:
        return [asdict(record) for record in records]
