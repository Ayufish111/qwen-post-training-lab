"""R0/R1/R2 的统一 RLVR 训练入口。

本文件只把三层职责接起来：
1. Transformers/PEFT 负责加载 Qwen3，并为三组创建同配置的新 LoRA；
2. TRL 负责 rollout、组内 advantage、policy loss、反向传播和 checkpoint；
3. 本项目负责 Multi-IF checker、奖励明细以及 R2 的约束感知采样。

真正的 GPU 代码只在通过 preflight 后导入，Windows 本机可以安全执行 `--preflight-only`。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.constraint_sampler import ConstraintSampler
from src.rlvr_contract import (
    EXPERIMENT_IDS,
    RewardBatchAdapter,
    load_algorithm_contract,
    load_training_pool,
    to_trl_rows,
    validate_rlvr_config,
)
from src.rlvr_rewards import RewardCalculator, group_has_zero_reward_variance

DEFAULT_CONFIG = PROJECT_DIR / "configs" / "rlvr.yaml"
LOCKED_TRL_VERSION = "1.9.2"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_steps(output_dir: Path) -> list[int]:
    if not output_dir.exists():
        return []
    result = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            if path.is_dir():
                result.append(int(path.name.split("-", 1)[1]))
        except (IndexError, ValueError):
            pass
    return sorted(result)


def validate_output_state(output_dir: Path, resume: bool) -> int | None:
    """防止误覆盖已有实验，并返回最新 checkpoint。"""
    steps = checkpoint_steps(output_dir)
    latest = steps[-1] if steps else None
    if resume and latest is None:
        raise ValueError(f"No checkpoint found under {output_dir}")
    if not resume and latest is not None:
        raise ValueError(f"Existing checkpoint-{latest} found under {output_dir}; use --resume")
    if not resume and output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Non-empty output directory has no checkpoint: {output_dir}")
    return latest


def select_smoke_rows(
    raw_rows: list[dict[str, Any]],
    trl_rows: list[dict[str, Any]],
    smoke_prompt_ids: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按配置顺序选择 smoke 题，正式训练不会调用这个函数。"""
    minimum_smoke_ids = min(4, len(raw_rows))
    if len(smoke_prompt_ids) < minimum_smoke_ids or len(smoke_prompt_ids) != len(set(smoke_prompt_ids)):
        raise ValueError(f"rlvr.smoke_prompt_ids must contain at least {minimum_smoke_ids} unique ids")
    raw_by_id = {str(row["id"]): row for row in raw_rows}
    trl_by_id = {str(row["row_id"]): row for row in trl_rows}
    missing = [row_id for row_id in smoke_prompt_ids if row_id not in raw_by_id or row_id not in trl_by_id]
    if missing:
        raise ValueError(f"smoke_prompt_ids are missing from the training pool: {missing}")
    return (
        [raw_by_id[row_id] for row_id in smoke_prompt_ids],
        [trl_by_id[row_id] for row_id in smoke_prompt_ids],
    )


def build_preflight(config: dict[str, Any], experiment_id: str, smoke: bool, resume: bool):
    validate_rlvr_config(config)
    contract = load_algorithm_contract(config, experiment_id)
    rlvr = config["rlvr"]
    if rlvr.get("blocked_until_t1_v2_generation_gate"):
        raise RuntimeError(
            "RLVR is blocked until the T1_v2 free-generation structure gate passes"
        )
    train_input = resolve_path(rlvr["train_input"])
    initial_adapter_value = rlvr.get("initial_adapter")
    initial_adapter = resolve_path(initial_adapter_value) if initial_adapter_value else None
    output_dir = resolve_path(rlvr["output_root"]) / (f"{experiment_id}_smoke" if smoke else experiment_id)
    if initial_adapter is not None:
        required = [initial_adapter / "adapter_config.json", initial_adapter / "adapter_model.safetensors"]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Initial adapter is incomplete: {missing}")
    rows = load_training_pool(train_input, expected_rows=int(config["data"]["train_size"]))
    latest = validate_output_state(output_dir, resume)
    per_step_sequences = int(rlvr["per_device_train_batch_size"]) * int(rlvr["gradient_accumulation_steps"])
    trl_rows = to_trl_rows(rows)
    active_rows = rows
    if smoke:
        active_rows, trl_rows = select_smoke_rows(rows, trl_rows, list(rlvr["smoke_prompt_ids"]))
    manifest = {
        "schema_version": "rlvr_preflight/v2",
        "experiment": experiment_id,
        "algorithm": asdict(contract),
        "dapo_consistency": (
            "not_applicable"
            if experiment_id == "R0"
            else "trl_native_dapo_loss_clip_higher_truncation_mask_plus_project_soft_overlong"
        ),
        "common_zero_variance_policy": rlvr["zero_variance_policy"],
        "zero_variance_abort_patience": int(rlvr["zero_variance_abort_patience"]),
        "zero_variance_tolerance": float(rlvr["zero_variance_tolerance"]),
        "train_input": str(train_input),
        "train_input_sha256": sha256_file(train_input),
        "train_rows": len(rows),
        "initialization": "existing_adapter" if initial_adapter is not None else rlvr["initialization"],
        "initial_adapter": str(initial_adapter) if initial_adapter is not None else None,
        "initial_adapter_sha256": sha256_file(initial_adapter / "adapter_model.safetensors") if initial_adapter is not None else None,
        "output_dir": str(output_dir),
        "smoke": smoke,
        "active_train_rows": len(active_rows),
        "smoke_selection_policy": rlvr["smoke_selection_policy"] if smoke else None,
        "smoke_prompt_ids": [row["id"] for row in active_rows] if smoke else [],
        # 保留旧 preflight manifest 的 5-step 约定；真实 smoke 运行步数单独记录为 2，
        # 这样既兼容已有审计文件，又避免付费机器上无必要地跑满 5 步。
        "requested_steps": 5 if smoke else int(rlvr["train_steps"]),
        "runtime_steps": 2 if smoke else int(rlvr["train_steps"]),
        "resume": resume,
        "resume_checkpoint_step": latest,
        "rollout_backend": rlvr["rollout_backend"],
        "rollout_enable_thinking": bool(rlvr["rollout_enable_thinking"]),
        "completions_per_prompt": int(rlvr["completions_per_prompt"]),
        "completion_sequences_per_optimizer_step": per_step_sequences,
        "prompt_groups_per_optimizer_step": per_step_sequences // int(rlvr["completions_per_prompt"]),
        "untouched_test_used": False,
    }
    # 保持原有 preflight API 的二元返回值，训练实现需要的原始行在 run_training 中再次读取。
    return manifest, trl_rows


class WeightedRepeatSampler:
    """TRL RepeatSampler 的单卡加权版本，始终按组连续重复 4 次。

    每个 generation batch 先按当前 CA 权重抽取若干个不同 prompt，再将每个索引连续
    重复 `num_generations` 次。权重函数在每个 batch 开始时重新读取，因此 reward 更新
    后无需重建 DataLoader。
    """

    def __init__(self, data_source, mini_repeat_count: int, batch_size: int, repeat_count: int, weights_fn, seed: int, shuffle: bool = True):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.weights_fn = weights_fn
        self.seed = seed
        self.shuffle = shuffle
        self.generator = None

    def __iter__(self):
        import torch

        generator = torch.Generator()
        generator.manual_seed(self.seed)
        total_batches = len(self.data_source) // self.batch_size
        for _ in range(total_batches):
            weights = torch.as_tensor(self.weights_fn(), dtype=torch.float32)
            if weights.numel() != len(self.data_source) or not torch.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError("CA sampler returned invalid weights")
            # 一个 batch 内不重复 prompt，保证每组都正好有 num_generations 个回答。
            if self.shuffle:
                chosen = torch.multinomial(weights, self.batch_size, replacement=False, generator=generator).tolist()
            else:
                chosen = list(range(self.batch_size))
            for _ in range(self.repeat_count):
                for index in chosen:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self):
        full_chunks = (len(self.data_source) // self.batch_size) * self.batch_size
        return full_chunks * self.mini_repeat_count * self.repeat_count


def build_grpo_kwargs(config: dict[str, Any], experiment_id: str, output_dir: Path, smoke: bool, bf16: bool) -> dict[str, Any]:
    """把冻结 YAML 显式映射到 TRL 参数，供测试锁定 R0/R1/R2 的归因边界。"""
    rlvr = config["rlvr"]
    contract = load_algorithm_contract(config, experiment_id)
    return {
        "output_dir": str(output_dir),
        "max_steps": 2 if smoke else int(rlvr["train_steps"]),
        "per_device_train_batch_size": int(rlvr["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(rlvr["gradient_accumulation_steps"]),
        "learning_rate": float(rlvr["learning_rate"]),
        "num_generations": int(rlvr["completions_per_prompt"]),
        "max_completion_length": int(rlvr["maximum_completion_tokens"]),
        "temperature": float(rlvr["temperature"]),
        "top_p": float(rlvr["top_p"]),
        "chat_template_kwargs": {
            "enable_thinking": bool(rlvr["rollout_enable_thinking"])
        },
        "epsilon": float(rlvr["symmetric_clip_epsilon"]),
        "epsilon_high": float(rlvr["clip_higher_epsilon_high"] if contract.clip_higher else rlvr["symmetric_clip_epsilon"]),
        "loss_type": "dapo" if contract.token_level_loss else "grpo",
        "mask_truncated_completions": bool(contract.token_level_loss),
        "beta": float(rlvr["kl_beta"]),
        "remove_unused_columns": False,
        "gradient_checkpointing": True,
        "optim": "paged_adamw_8bit",
        "bf16": bf16,
        "fp16": not bf16,
        "logging_steps": 1,
        "save_strategy": "steps",
        "save_steps": 1 if smoke else int(rlvr["save_steps"]),
        "save_total_limit": int(rlvr["save_total_limit"]),
        "report_to": "none",
        "seed": int(config["seed"]),
        "data_seed": int(config["seed"]),
        "log_completions": False,
    }


def validate_live_trl_binding(grpo_config_class: type, grpo_trainer_class: type) -> str:
    """验证当前进程确实运行在审计过的 TRL 版本和接口上。"""
    from importlib.metadata import version

    installed = version("trl")
    if installed != LOCKED_TRL_VERSION:
        raise RuntimeError(f"TRL version mismatch: expected {LOCKED_TRL_VERSION}, found {installed}")
    config_fields = set(getattr(grpo_config_class, "__dataclass_fields__", {}))
    required_fields = {"num_generations", "max_completion_length", "epsilon", "epsilon_high", "loss_type", "mask_truncated_completions"}
    missing_fields = required_fields - config_fields
    required_methods = {"_get_train_sampler", "_save_checkpoint", "_load_optimizer_and_scheduler"}
    missing_methods = {name for name in required_methods if not callable(getattr(grpo_trainer_class, name, None))}
    if missing_fields or missing_methods:
        raise RuntimeError(f"TRL binding mismatch; missing fields={sorted(missing_fields)}, methods={sorted(missing_methods)}")
    return installed


def load_official_ifeval() -> Any:
    checker_dir = PROJECT_DIR / "third_party" / "Multi-IF"
    sys.path.insert(0, str(checker_dir.resolve()))
    ifeval = importlib.import_module("ifeval")
    local_nltk_data = PROJECT_DIR / "nltk_data"
    if local_nltk_data.exists() and str(local_nltk_data.resolve()) not in ifeval.nltk.data.path:
        ifeval.nltk.data.path.insert(0, str(local_nltk_data.resolve()))
    return ifeval


def trainable_parameter_sha256(model: Any) -> tuple[str, int]:
    """精确哈希所有可训练 LoRA 参数，用于证明 smoke 确实更新了权重。"""
    import torch

    digest = hashlib.sha256()
    tensor_count = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        tensor = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        tensor_count += 1
    if tensor_count == 0:
        raise RuntimeError("Model has no trainable parameters")
    return digest.hexdigest(), tensor_count


def advantages_have_learning_signal(values: Any, tolerance: float) -> bool:
    """只要当前 generation batch 至少有一个非零 advantage，就允许执行更新。"""
    if hasattr(values, "detach"):
        values = values.detach().float().cpu().tolist()
    flat: list[float] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            flat.extend(float(item) for item in value)
        else:
            flat.append(float(value))
    return any(abs(value) > tolerance for value in flat)


def evaluate_smoke_health(
    stats: list[dict[str, Any]],
    parameter_hash_before: str,
    parameter_hash_after: str,
) -> dict[str, Any]:
    """汇总并验收两步 GPU smoke，不改 reward 或 rollout 样本。"""
    if not stats:
        raise RuntimeError("Smoke failed: no reward statistics were written")
    summary = {
        "trainable_parameter_sha256_before": parameter_hash_before,
        "trainable_parameter_sha256_after": parameter_hash_after,
        "trainable_parameters_changed": parameter_hash_after != parameter_hash_before,
        "smoke_checker_errors": max(item["checker_errors"] for item in stats),
        "smoke_has_nonzero_variance_group": any(
            item["zero_variance_group_ratio"] < 1.0 for item in stats
        ),
        "smoke_parse_ok_ratio": sum(item["parse_ok_ratio"] for item in stats) / len(stats),
        "smoke_completion_at_limit_ratio": sum(
            item["completion_at_limit_ratio"] for item in stats
        ) / len(stats),
    }
    if summary["smoke_checker_errors"] != 0:
        raise RuntimeError("Smoke failed: official checker raised an error")
    if not summary["smoke_has_nonzero_variance_group"]:
        raise RuntimeError("Smoke failed: every prompt group still has zero reward variance")
    if summary["smoke_parse_ok_ratio"] < 1.0:
        raise RuntimeError("Smoke failed: at least one completion did not preserve thinking/answer structure")
    if summary["smoke_completion_at_limit_ratio"] >= 1.0:
        raise RuntimeError("Smoke failed: every completion reached the generation limit")
    if not summary["trainable_parameters_changed"]:
        raise RuntimeError("Smoke failed: LoRA trainable parameters did not change")
    return summary


def decode_reward_completions(
    tokenizer: Any,
    completions: list[Any],
    completion_ids: Any | None,
) -> list[Any]:
    """Reward 优先使用保留 `</think>` 和 EOS 的原始 token 解码。"""
    if completion_ids is None:
        return completions
    return tokenizer.batch_decode(completion_ids, skip_special_tokens=False)


def configure_qwen_chat_termination(tokenizer: Any) -> dict[str, Any]:
    """让聊天生成在 `<|im_end|>` 结束，同时保留独立的 padding token。"""
    chat_end_token = "<|im_end|>"
    text_end_token = "<|endoftext|>"
    chat_end_id = tokenizer.convert_tokens_to_ids(chat_end_token)
    unknown_id = getattr(tokenizer, "unk_token_id", None)
    if not isinstance(chat_end_id, int) or chat_end_id < 0 or chat_end_id == unknown_id:
        raise RuntimeError(f"tokenizer does not define {chat_end_token}")
    if tokenizer.pad_token_id is None:
        text_end_id = tokenizer.convert_tokens_to_ids(text_end_token)
        if not isinstance(text_end_id, int) or text_end_id < 0 or text_end_id == unknown_id:
            raise RuntimeError("tokenizer has neither a pad token nor <|endoftext|>")
        tokenizer.pad_token = text_end_token
    tokenizer.eos_token = chat_end_token
    if tokenizer.eos_token_id != chat_end_id:
        raise RuntimeError("failed to bind tokenizer EOS to <|im_end|>")
    return {
        "eos_token": tokenizer.eos_token,
        "eos_token_id": int(tokenizer.eos_token_id),
        "pad_token": tokenizer.pad_token,
        "pad_token_id": int(tokenizer.pad_token_id),
    }


def run_training(args, config: dict[str, Any], manifest: dict[str, Any], trl_rows: list[dict[str, Any]]):
    """在 AutoDL 执行一次真实 TRL 训练；R0/R1/R2 只由 contract 开关区分。"""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from trl import GRPOConfig, GRPOTrainer

    manifest["trl_version"] = validate_live_trl_binding(GRPOConfig, GRPOTrainer)

    rlvr = config["rlvr"]
    contract = load_algorithm_contract(config, args.experiment)
    zero_variance_tolerance = float(rlvr["zero_variance_tolerance"])
    zero_variance_abort_patience = int(rlvr["zero_variance_abort_patience"])
    manifest["common_zero_variance_policy"] = rlvr["zero_variance_policy"]
    manifest["zero_variance_abort_patience"] = zero_variance_abort_patience
    manifest["zero_variance_tolerance"] = zero_variance_tolerance
    full_raw_rows = load_training_pool(resolve_path(rlvr["train_input"]), expected_rows=int(config["data"]["train_size"]))
    raw_rows = full_raw_rows
    if args.smoke:
        raw_rows, expected_trl_rows = select_smoke_rows(
            full_raw_rows, to_trl_rows(full_raw_rows), list(rlvr["smoke_prompt_ids"])
        )
        if [row["row_id"] for row in trl_rows] != [row["row_id"] for row in expected_trl_rows]:
            raise RuntimeError("Smoke rows changed between preflight and training")
    model_path = args.model or config["model"]["primary"]
    adapter_value = rlvr.get("initial_adapter")
    adapter_path = resolve_path(adapter_value) if adapter_value else None
    output_dir = Path(manifest["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")

    if not torch.cuda.is_available():
        raise RuntimeError("RLVR requires CUDA; preflight itself remains CPU-safe.")
    # LoRA A 矩阵是随机初始化的。必须在建模前固定种子，才能让 R0/R1/R2
    # 不只是“配置相同”，而是从完全相同的可训练权重出发。
    initialization_seed = int(config["seed"])
    set_seed(initialization_seed)
    manifest["initialization_seed"] = initialization_seed
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=args.local_files_only, trust_remote_code=False, padding_side="left")
    manifest["generation_tokens"] = configure_qwen_chat_termination(tokenizer)

    # TRL 1.9.2 已移除 max_prompt_length 参数，所以在进入昂贵 rollout 前主动检查硬上限。
    prompt_lengths = []
    for row in trl_rows:
        rendered = tokenizer.apply_chat_template(
            row["prompt"], tokenize=True, add_generation_prompt=True,
            enable_thinking=bool(rlvr["rollout_enable_thinking"]),
        )
        prompt_lengths.append(len(rendered))
    too_long = [
        (trl_rows[index]["row_id"], length)
        for index, length in enumerate(prompt_lengths)
        if length > int(rlvr["maximum_prompt_tokens"])
    ]
    if too_long:
        raise RuntimeError(
            f"{len(too_long)} prompts exceed maximum_prompt_tokens="
            f"{rlvr['maximum_prompt_tokens']}; first rows: {too_long[:5]}"
        )
    manifest["prompt_tokens_max"] = max(prompt_lengths)
    manifest["prompt_tokens_mean"] = sum(prompt_lengths) / len(prompt_lengths)

    ifeval = load_official_ifeval()
    unknown_instruction_ids = sorted({
        instruction_id for row in raw_rows for instruction_id in row["instruction_ids"]
        if instruction_id not in ifeval.INSTRUCTION_DICT
    })
    if unknown_instruction_ids:
        raise RuntimeError(f"Training pool contains unknown checker ids: {unknown_instruction_ids}")

    base = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=args.local_files_only, trust_remote_code=False, quantization_config=quantization, device_map={"": 0}, dtype=compute_dtype, low_cpu_mem_usage=True)
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    if adapter_path is not None:
        model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=True)
    else:
        qlora = rlvr["qlora"]
        model = get_peft_model(
            base,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=qlora["target_modules"],
                r=int(qlora["r"]),
                lora_alpha=int(qlora["lora_alpha"]),
                lora_dropout=float(qlora["lora_dropout"]),
                bias="none",
            ),
        )
    parameter_hash_before, trainable_tensor_count = trainable_parameter_sha256(model)
    manifest["trainable_parameter_sha256_before"] = parameter_hash_before
    manifest["trainable_tensor_count"] = trainable_tensor_count

    checker = RewardCalculator(ifeval)
    ca = ConstraintSampler(**{
        "ema_beta": config["constraint_sampling"]["ema_beta"],
        "mixture_lambda": config["constraint_sampling"]["uniform_mixture_lambda"],
        "min_sampling_weight": config["constraint_sampling"]["min_sampling_weight"],
        "max_sampling_weight": config["constraint_sampling"]["max_sampling_weight"],
        "update_steps": config["constraint_sampling"]["sampling_weight_update_steps"],
        "fast_ema_beta": config["constraint_sampling"]["fast_ema_beta"],
        "stagnation_tolerance": config["constraint_sampling"]["stagnation_tolerance"],
        "stagnation_patience": config["constraint_sampling"]["stagnation_patience"],
        "stagnation_scale": config["constraint_sampling"]["stagnation_scale"],
    })
    stats_path = output_dir / "rlvr_stats.jsonl"
    runtime_state = {"consecutive_zero_signal_batches": 0}
    seen_row_ids: set[str] = set()
    if args.resume and stats_path.is_file():
        for line in stats_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                seen_row_ids.update(json.loads(line).get("sampled_row_ids", []))

    def token_count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    adapter = RewardBatchAdapter(checker, contract, token_count, int(rlvr["maximum_completion_tokens"]), int(rlvr["overlong_buffer_tokens"]))

    def reward_func(completions, instruction_ids, constraint_kwargs, row_id, **kwargs):
        completion_ids = kwargs.get("completion_ids")
        # TRL 的普通 conversational 分支使用 skip_special_tokens=True 解码，会把 Qwen3 的
        # </think> 特殊 token 删除。奖励必须从原始 token id 重新解码，否则 thinking 与
        # 最终回答无法拆开，所有正常生成都会被误判为 0 分。
        reward_completions = decode_reward_completions(
            tokenizer, completions, completion_ids
        )
        rewards = adapter(
            reward_completions,
            instruction_ids,
            constraint_kwargs,
            row_id,
            **kwargs,
        )
        seen_row_ids.update(str(value) for value in row_id)
        records = adapter.pop_records()
        if records:
            if contract.constraint_aware_sampling:
                ca.update_batch_pass_rates(adapter.category_pass_rates(records))
            groups = [records[i:i + int(rlvr["completions_per_prompt"])] for i in range(0, len(records), int(rlvr["completions_per_prompt"]))]
            category_rates = adapter.category_pass_rates(records)
            payload = {"step": int(kwargs.get("trainer_state").global_step) if kwargs.get("trainer_state") else None, "sampled_row_ids": list(dict.fromkeys(str(value) for value in row_id)), "reward_mean": sum(rewards) / len(rewards), "rewards": rewards, "parse_ok_ratio": sum(record.parse_ok for record in records) / len(records), "reasoning_missing_ratio": sum(record.reasoning_missing for record in records) / len(records), "answer_only_fallback_ratio": sum(record.answer_only_fallback for record in records) / len(records), "completion_at_limit_ratio": sum(record.completion_tokens >= int(rlvr["maximum_completion_tokens"]) for record in records) / len(records), "category_pass_rates": category_rates, "zero_variance_group_ratio": sum(group_has_zero_reward_variance([r.reward for r in group], zero_variance_tolerance) for group in groups) / max(len(groups), 1), "checker_errors": checker.checker_errors, "unique_prompts_seen": len(seen_row_ids), "unique_prompt_coverage": len(seen_row_ids) / len(full_raw_rows)}
            if args.smoke:
                payload["decoded_samples"] = [
                    text[:1000] for text in reward_completions[:2]
                ]
            if contract.constraint_aware_sampling:
                payload["sampler_state"] = ca.state_dict()
            with stats_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return rewards

    dataset = Dataset.from_list(trl_rows)
    class ProjectGRPOTrainer(GRPOTrainer):
        def _generate_and_score_completions(self, inputs):
            if not self.model.training:
                return super()._generate_and_score_completions(inputs)
            # 不按 reward 结果筛选或重生成回答；零方差批仍按原始 GRPO 语义得到 0 advantage。
            # 这里只监控连续空更新，达到门限便停机，避免付费 GPU 长时间做无效计算。
            output = super()._generate_and_score_completions(inputs)
            if advantages_have_learning_signal(output["advantages"], zero_variance_tolerance):
                runtime_state["consecutive_zero_signal_batches"] = 0
                self._metrics["train"]["runtime/consecutive_zero_signal_batches"].append(0.0)
                return output
            runtime_state["consecutive_zero_signal_batches"] += 1
            consecutive = runtime_state["consecutive_zero_signal_batches"]
            self._metrics["train"]["runtime/consecutive_zero_signal_batches"].append(float(consecutive))
            if consecutive < zero_variance_abort_patience:
                return output
            failure = {
                "schema_version": "rlvr_zero_variance_abort/v1",
                "experiment": args.experiment,
                "global_step": int(self.state.global_step),
                "consecutive_zero_signal_batches": consecutive,
                "policy": rlvr["zero_variance_policy"],
                "message": "training batches were accepted unchanged but repeatedly had zero advantage",
            }
            (output_dir / "zero_variance_abort.json").write_text(
                json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            raise RuntimeError(
                f"All prompt groups had zero reward variance for {consecutive} consecutive batches; "
                "training aborted without reward manipulation to protect GPU budget"
            )

        def _get_train_sampler(self, dataset=None):
            if not contract.constraint_aware_sampling and not args.smoke:
                return super()._get_train_sampler(dataset)
            source = dataset if dataset is not None else self.train_dataset
            generation_batch = self.args.generation_batch_size // self.args.num_generations
            repeat_count = self.num_iterations * self.args.steps_per_generation
            weights_fn = (
                (lambda: [1.0] * len(source))
                if args.smoke and not contract.constraint_aware_sampling
                else lambda: ca.weights_for_step(raw_rows, self.state.global_step)
            )
            return WeightedRepeatSampler(
                source,
                self.args.num_generations,
                generation_batch,
                repeat_count,
                weights_fn,
                int(self.args.seed),
                shuffle=not args.smoke,
            )

        def _save_checkpoint(self, model, trial):
            super()._save_checkpoint(model, trial)
            if contract.constraint_aware_sampling and self.args.should_save:
                checkpoint = Path(self._get_output_dir(trial=trial)) / f"checkpoint-{self.state.global_step}"
                (checkpoint / "ca_sampler_state.json").write_text(
                    json.dumps(ca.state_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
                )

        def _load_optimizer_and_scheduler(self, checkpoint):
            super()._load_optimizer_and_scheduler(checkpoint)
            if contract.constraint_aware_sampling and checkpoint:
                state_path = Path(checkpoint) / "ca_sampler_state.json"
                if not state_path.is_file():
                    raise RuntimeError(f"R2 checkpoint is missing sampler state: {state_path}")
                ca.load_state_dict(json.loads(state_path.read_text(encoding="utf-8")))

    trainer_args = GRPOConfig(**build_grpo_kwargs(config, args.experiment, output_dir, args.smoke, compute_dtype == torch.bfloat16))
    trainer = ProjectGRPOTrainer(model=model, reward_funcs=reward_func, args=trainer_args, train_dataset=dataset, processing_class=tokenizer)
    train_result = trainer.train(resume_from_checkpoint=True if args.resume else None)
    if args.smoke:
        parameter_hash_after, _ = trainable_parameter_sha256(model)
        stats = [
            json.loads(line) for line in stats_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        manifest.update(
            evaluate_smoke_health(stats, parameter_hash_before, parameter_hash_after)
        )
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    final_adapter = output_dir / "final_adapter"
    trainer.save_model(str(final_adapter))
    tokenizer.save_pretrained(str(final_adapter))
    if args.smoke:
        del trainer, model, base
        gc.collect()
        torch.cuda.empty_cache()
        reload_base = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=args.local_files_only, trust_remote_code=False, quantization_config=quantization, device_map={"": 0}, dtype=compute_dtype, low_cpu_mem_usage=True)
        reloaded = PeftModel.from_pretrained(reload_base, str(final_adapter))
        print("smoke adapter reload: OK", type(reloaded).__name__)
    manifest["status"] = "completed"
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"final adapter saved to: {final_adapter}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=EXPERIMENT_IDS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model")
    parser.add_argument("--initial-adapter", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.resume:
        parser.error("--smoke and --resume cannot be used together")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.initial_adapter is not None:
        config["rlvr"]["initial_adapter"] = str(args.initial_adapter)
        config["rlvr"]["initialization"] = "existing_adapter"
    manifest, trl_rows = build_preflight(config, args.experiment, args.smoke, args.resume)
    manifest["model"] = args.model or config["model"]["primary"]
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.preflight_only:
        print("RLVR preflight: OK")
        return
    run_training(args, config, manifest, trl_rows)


if __name__ == "__main__":
    main()
