"""训练 R0/R1/R2 共用的 T1 Thinking cold-start QLoRA adapter。

T1_v2 在七类 Transformer 线性层之外只训练 thinking/EOS 三个 tied token 行，
并在 token-level loss 中适度提高这三个结构位置的权重。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from datasets import load_from_disk
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.rlvr_rewards import parse_reasoning_answer


DEFAULT_CONFIG = PROJECT_DIR / "configs" / "rlvr.yaml"


def resolve_structure_token_ids(tokenizer, token_strings: list[str]) -> list[int]:
    """把配置中的结构 token 严格解析为单个 token id。"""
    result: list[int] = []
    for token in token_strings:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"Structure token must encode to one id: {token!r} -> {encoded}")
        token_id = int(encoded[0])
        if tokenizer.convert_ids_to_tokens(token_id) != token:
            raise RuntimeError(f"Structure token round-trip failed: {token!r} -> {token_id}")
        result.append(token_id)
    if len(result) != len(set(result)):
        raise RuntimeError("Structure token ids must be unique")
    return result


def configure_qwen_training_tokens(tokenizer) -> dict[str, int]:
    """训练和自由生成统一使用 `<|im_end|>` 作为 EOS。"""
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    text_end_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if not isinstance(im_end_id, int) or not isinstance(text_end_id, int):
        raise RuntimeError("Qwen tokenizer is missing required end tokens")
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.pad_token = "<|endoftext|>"
    if tokenizer.eos_token_id != im_end_id or tokenizer.pad_token_id != text_end_id:
        raise RuntimeError("failed to bind Qwen EOS/PAD tokens")
    return {"eos_token_id": im_end_id, "pad_token_id": text_end_id}


def find_audit_rows(tokenized, row_ids: list[str]) -> list[dict]:
    """从冻结 cache 中按顺序取出 generation gate 样本。"""
    found = {}
    for split_name in ("t1_train", "t1_validation"):
        for row in tokenized[split_name]:
            if row["id"] in row_ids:
                found[row["id"]] = row
    missing = [row_id for row_id in row_ids if row_id not in found]
    if missing:
        raise RuntimeError(f"generation gate rows are missing from T1 cache: {missing}")
    return [found[row_id] for row_id in row_ids]


def select_smoke_rows(dataset, sample_count: int, offset: int, required_ids: list[str]):
    """固定选取多样 smoke 样本，并优先保留当前 split 中的门禁样本。"""
    if sample_count <= 0:
        raise ValueError("smoke sample count must be positive")
    if sample_count > len(dataset):
        raise ValueError(
            f"smoke sample count {sample_count} exceeds split size {len(dataset)}"
        )
    if offset < 0:
        raise ValueError("smoke offset must be non-negative")

    row_ids = list(dataset["id"])
    id_to_index = {row_id: index for index, row_id in enumerate(row_ids)}
    selected = [id_to_index[row_id] for row_id in required_ids if row_id in id_to_index]
    if len(selected) > sample_count:
        raise ValueError("smoke sample count is smaller than required gate rows")

    start = offset % len(dataset)
    rotated = list(range(start, len(dataset))) + list(range(0, start))
    selected_set = set(selected)
    for index in rotated:
        if index in selected_set:
            continue
        selected.append(index)
        selected_set.add(index)
        if len(selected) == sample_count:
            break
    return dataset.select(selected)


def inspect_generated_structure(
    generated: list[int], think_open_id: int, think_close_id: int, eos_token_id: int
) -> dict:
    """检查唯一 thinking 块、先后顺序和自然 EOS，不接受只出现过一次的宽松判断。"""
    open_positions = [i for i, token_id in enumerate(generated) if token_id == think_open_id]
    close_positions = [i for i, token_id in enumerate(generated) if token_id == think_close_id]
    eos_positions = [i for i, token_id in enumerate(generated) if token_id == eos_token_id]
    starts_with_think_open = bool(generated) and generated[0] == think_open_id
    ordered = (
        len(open_positions) == 1
        and len(close_positions) == 1
        and len(eos_positions) >= 1
        and open_positions[0] < close_positions[0] < eos_positions[0]
    )
    return {
        "think_open_count": len(open_positions),
        "think_close_count": len(close_positions),
        "starts_with_think_open": starts_with_think_open,
        "structure_order_ok": ordered,
        "natural_im_end": bool(eos_positions),
        "strict_structure_ok": starts_with_think_open and ordered,
    }


def run_free_generation_gate(model, tokenizer, rows: list[dict], max_new_tokens: int) -> dict:
    """使用重载后 adapter 执行贪心生成，不使用教师 completion。"""
    model.eval()
    eos_token_id = int(tokenizer.eos_token_id)
    pad_token_id = int(tokenizer.pad_token_id)
    think_open_id = int(tokenizer.convert_tokens_to_ids("<think>"))
    think_close_id = int(tokenizer.convert_tokens_to_ids("</think>"))
    input_device = model.get_input_embeddings().weight.device
    records = []
    for row in rows:
        prompt_length = int(row["prompt_length"])
        prompt_ids = torch.tensor(
            [row["input_ids"][:prompt_length]], dtype=torch.long, device=input_device
        )
        attention_mask = torch.ones_like(prompt_ids)
        with torch.inference_mode():
            output = model.generate(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )
        generated = output[0, prompt_length:].tolist()
        structure = inspect_generated_structure(
            generated, think_open_id, think_close_id, eos_token_id
        )
        if structure["natural_im_end"]:
            generated = generated[: generated.index(eos_token_id) + 1]
        decoded = tokenizer.decode(generated, skip_special_tokens=False)
        parsed = parse_reasoning_answer(decoded)
        records.append(
            {
                "row_id": row["id"],
                "generated_tokens": len(generated),
                "parse_ok": parsed is not None,
                "decoded_preview": decoded[:1000],
                **structure,
            }
        )
    passed = all(
        record["strict_structure_ok"]
        and record["natural_im_end"]
        and record["parse_ok"]
        for record in records
    )
    return {
        "schema_version": "t1_free_generation_gate/v1",
        "passed": passed,
        "max_new_tokens": max_new_tokens,
        "records": records,
    }


def structure_weighted_causal_loss(
    outputs,
    labels: torch.Tensor,
    structure_token_ids: list[int],
    structure_multiplier: float,
) -> torch.Tensor:
    """在不展开整张 fp32 词表的前提下，精确提高结构位置的相对权重。"""
    if structure_multiplier < 1.0:
        raise ValueError("structure_multiplier must be at least 1")
    if outputs.loss is None:
        raise RuntimeError("model did not return the base causal loss")
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = labels[:, 1:].contiguous()
    structure_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
    for token_id in structure_token_ids:
        structure_mask |= shift_labels.eq(token_id)
    if not bool(structure_mask.any()):
        raise RuntimeError("batch has no supervised structure tokens")
    selected_logits = shift_logits[structure_mask].float()
    selected_labels = shift_labels[structure_mask]
    structure_loss = torch.nn.functional.cross_entropy(
        selected_logits, selected_labels, reduction="mean"
    )
    active_count = shift_labels.ne(-100).sum().to(outputs.loss.dtype)
    structure_count = structure_mask.sum().to(outputs.loss.dtype)
    extra_weight = structure_multiplier - 1.0
    weighted_sum = outputs.loss * active_count + structure_loss * (
        extra_weight * structure_count
    )
    total_weight = active_count + extra_weight * structure_count
    return weighted_sum / total_weight


def lora_b_update_audit(model) -> dict:
    """检查 Transformer LoRA 与结构 token 增量是否真正更新。"""
    tensors = []
    token_deltas = []
    for name, parameter in model.named_parameters():
        max_abs = float(parameter.detach().float().abs().max().cpu())
        if "lora_B" in name:
            tensors.append({"name": name, "max_abs": max_abs})
        elif "trainable_tokens_delta" in name:
            token_deltas.append({"name": name, "max_abs": max_abs})
    nonzero = [item for item in tensors if item["max_abs"] > 0.0]
    token_delta_nonzero = any(
        item["max_abs"] > 0.0 for item in token_deltas
    )
    return {
        "schema_version": "t1_adapter_update_audit/v2",
        "lora_b_tensor_count": len(tensors),
        "nonzero_lora_b_tensor_count": len(nonzero),
        "structure_token_delta_tensor_count": len(token_deltas),
        "structure_token_delta_nonzero": token_delta_nonzero,
        "structure_token_updates": token_deltas,
        "largest_updates": sorted(nonzero, key=lambda item: item["max_abs"], reverse=True)[:10],
    }


class StructureWeightedTrainer(Trainer):
    """T1_v2 Trainer：仅替换 loss 聚合，其余训练流程保持 Hugging Face 默认实现。"""

    def __init__(
        self, *args, structure_token_ids: list[int], structure_multiplier: float, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.structure_token_ids = list(structure_token_ids)
        self.structure_multiplier = float(structure_multiplier)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        outputs = model(**inputs)
        loss = structure_weighted_causal_loss(
            outputs,
            labels,
            self.structure_token_ids,
            self.structure_multiplier,
        )
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir=None, state_dict=None):
        """只保存 LoRA 与结构 token 增量，不保存整张词表权重。"""
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
        if not isinstance(model, PeftModel):
            raise RuntimeError("T1_v2 expected a PeftModel during checkpoint save")
        model.save_pretrained(
            output_dir,
            state_dict=state_dict,
            safe_serialization=getattr(self.args, "save_safetensors", True),
            save_embedding_layers=False,
        )
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        torch.save(self.args, Path(output_dir) / "training_args.bin")


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="T1_v2", choices=["T1", "T1_v2"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", help="Remote model id or local Base snapshot path.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run 2 train rows and 2 validation rows in T1_v2_smoke.",
    )
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=5,
        help="Number of optimizer steps for smoke; formal training ignores this value.",
    )
    parser.add_argument(
        "--smoke-gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation used only by smoke; set to the formal value for a faithful batch-size diagnostic.",
    )
    parser.add_argument(
        "--smoke-train-samples",
        type=int,
        default=2,
        help="Number of fixed diverse training rows used only by smoke.",
    )
    parser.add_argument(
        "--smoke-eval-samples",
        type=int,
        default=2,
        help="Number of fixed validation rows used only by smoke.",
    )
    parser.add_argument(
        "--smoke-offset",
        type=int,
        default=0,
        help="Rotate the deterministic smoke pool without changing the frozen dataset.",
    )
    parser.add_argument(
        "--smoke-output-dir",
        type=Path,
        help="Separate output directory for a smoke cycle; formal training ignores it.",
    )
    parser.add_argument(
        "--initial-adapter",
        type=Path,
        help="Continue a smoke diagnostic from an existing T1_v2 adapter with a new optimizer.",
    )
    parser.add_argument(
        "--skip-generation-gate",
        action="store_true",
        help="Skip expensive free generation for a local weight-update diagnostic only.",
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    t1 = config["t1"]
    qlora = t1["qlora"]
    training = t1["training"]
    model_name = args.model or config["model"]["primary"]
    cache_dir = resolve_path(t1["cache"])
    repair_structure = args.experiment == "T1_v2"
    output_key = "output_dir" if repair_structure else "legacy_output_dir"
    output_dir = resolve_path(training[output_key])
    if args.smoke:
        output_dir = (
            resolve_path(args.smoke_output_dir)
            if args.smoke_output_dir
            else output_dir.parent / f"{output_dir.name}_smoke"
        )
    if args.smoke and args.resume:
        parser.error("--smoke and --resume cannot be used together")
    if args.smoke_output_dir and not args.smoke:
        parser.error("--smoke-output-dir is allowed only with --smoke")
    if args.initial_adapter and not args.smoke:
        parser.error("--initial-adapter is allowed only with --smoke")
    if args.initial_adapter and args.resume:
        parser.error("--initial-adapter and --resume cannot be used together")
    if args.smoke_steps <= 0:
        parser.error("--smoke-steps must be positive")
    if args.smoke_train_samples <= 0 or args.smoke_eval_samples <= 0:
        parser.error("smoke sample counts must be positive")
    if args.smoke_gradient_accumulation_steps <= 0:
        parser.error("--smoke-gradient-accumulation-steps must be positive")
    if args.smoke_offset < 0:
        parser.error("--smoke-offset must be non-negative")
    if args.skip_generation_gate and not args.smoke:
        parser.error("--skip-generation-gate is allowed only with --smoke")

    if not cache_dir.exists():
        raise RuntimeError(
            f"T1 cache is missing: {cache_dir}. Run src/build_distill_dataset.py first."
        )
    last_checkpoint = get_last_checkpoint(str(output_dir)) if output_dir.exists() else None
    if args.resume and last_checkpoint is None:
        parser.error(f"No checkpoint found under {output_dir}")
    if not args.resume and last_checkpoint is not None:
        raise RuntimeError(
            f"Existing checkpoint found: {last_checkpoint}. Use --resume to continue, "
            "or move the old experiment directory before restarting."
        )
    if (
        not args.resume
        and output_dir.exists()
        and any(output_dir.iterdir())
        and last_checkpoint is None
    ):
        raise RuntimeError(
            f"Non-empty output directory has no resumable checkpoint: {output_dir}. "
            "Move it aside before starting a new run."
        )
    resume_checkpoint = last_checkpoint if args.resume else None

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=False,
        local_files_only=args.local_files_only,
    )
    tokenizer.padding_side = "right"
    generation_tokens = configure_qwen_training_tokens(tokenizer)
    structure_token_ids = resolve_structure_token_ids(
        tokenizer, list(qlora["structure_token_strings"])
    )

    tokenized = load_from_disk(str(cache_dir))
    required_splits = {"t1_train", "t1_validation"}
    missing = required_splits - set(tokenized)
    if missing:
        raise RuntimeError(f"T1 cache is missing splits: {sorted(missing)}")

    if not torch.cuda.is_available():
        raise RuntimeError("T1 QLoRA training requires an NVIDIA GPU.")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU preflight: {gpu.name}, {gpu.total_memory / 1024**3:.1f} GB")
    print("compute dtype:", compute_dtype)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=bool(qlora["load_in_4bit"]),
        bnb_4bit_quant_type=qlora["quant_type"],
        bnb_4bit_use_double_quant=bool(qlora["double_quant"]),
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=False,
        local_files_only=args.local_files_only,
        quantization_config=quantization_config,
        device_map={"": 0},
        dtype=compute_dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )
    initial_adapter = resolve_path(args.initial_adapter) if args.initial_adapter else None
    if initial_adapter:
        if not (initial_adapter / "adapter_config.json").is_file():
            raise RuntimeError(f"initial adapter is incomplete: {initial_adapter}")
        model = PeftModel.from_pretrained(
            model, str(initial_adapter), is_trainable=True
        )
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=(
                    qlora["target_modules"]
                    if repair_structure
                    else qlora["legacy_target_modules"]
                ),
                r=int(qlora["r"]),
                lora_alpha=int(qlora["lora_alpha"]),
                lora_dropout=float(qlora["lora_dropout"]),
                bias="none",
                trainable_token_indices=structure_token_ids if repair_structure else None,
                ensure_weight_tying=repair_structure,
            ),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    max_steps = args.smoke_steps if args.smoke else int(training["max_steps"])
    eval_steps = 1 if args.smoke else int(training["eval_steps"])
    save_steps = 1 if args.smoke else int(training["save_steps"])
    gradient_accumulation_steps = (
        args.smoke_gradient_accumulation_steps
        if args.smoke
        else int(training["gradient_accumulation_steps"])
    )
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training["per_device_eval_batch_size"]),
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=float(training["learning_rate"]),
        # 一步 smoke 配上线性 warmup 时唯一一步的 LR 会是 0。固定学习率只用于
        # 本地小步诊断；正式 200 步仍严格使用 YAML 中冻结的线性调度。
        lr_scheduler_type="constant" if args.smoke else training["lr_scheduler_type"],
        warmup_ratio=0.0 if args.smoke else float(training["warmup_ratio"]),
        optim="paged_adamw_8bit",
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        gradient_checkpointing=True,
        logging_steps=int(training["logging_steps"]),
        logging_first_step=True,
        # smoke 在训练后仍会显式 evaluate；中间评估和 optimizer checkpoint
        # 只会拖慢本地诊断。正式训练保持每 50 步评估、保存和选择最佳模型。
        eval_strategy="no" if args.smoke else "steps",
        eval_steps=eval_steps,
        save_strategy="no" if args.smoke else "steps",
        save_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=not args.smoke,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=int(config["seed"]),
        data_seed=int(config["seed"]),
        run_name=args.experiment,
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    train_dataset = tokenized["t1_train"]
    eval_dataset = tokenized["t1_validation"]
    if args.smoke:
        gate_ids = list(t1["generation_gate"]["audit_rows"])
        train_dataset = select_smoke_rows(
            train_dataset,
            args.smoke_train_samples,
            args.smoke_offset,
            gate_ids,
        )
        eval_dataset = select_smoke_rows(
            eval_dataset,
            args.smoke_eval_samples,
            args.smoke_offset,
            gate_ids,
        )
        smoke_manifest = {
            "schema_version": "t1_smoke_selection/v1",
            "initial_adapter": str(initial_adapter) if initial_adapter else None,
            "train_ids": list(train_dataset["id"]),
            "validation_ids": list(eval_dataset["id"]),
            "offset": args.smoke_offset,
            "optimizer_steps": max_steps,
            "gradient_accumulation_steps": gradient_accumulation_steps,
        }
        (output_dir / "smoke_manifest.json").write_text(
            json.dumps(smoke_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    trainer_class = StructureWeightedTrainer if repair_structure else Trainer
    trainer_kwargs = {}
    if repair_structure:
        trainer_kwargs.update(
            structure_token_ids=structure_token_ids,
            structure_multiplier=float(qlora["structure_token_loss_multiplier"]),
        )
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        **trainer_kwargs,
    )

    print("experiment:", args.experiment)
    print("model:", model_name)
    print("cache:", cache_dir)
    print("output:", output_dir)
    print("initial adapter:", initial_adapter)
    print("train samples:", len(train_dataset))
    print("validation samples:", len(eval_dataset))
    print("optimizer steps:", max_steps)
    print("structure repair:", repair_structure)
    print("structure token ids:", structure_token_ids)
    print("generation tokens:", generation_tokens)
    print(
        "structure token loss multiplier:",
        float(qlora["structure_token_loss_multiplier"]) if repair_structure else 1.0,
    )
    print(
        "effective batch size:",
        int(training["per_device_train_batch_size"])
        * gradient_accumulation_steps,
    )
    model.print_trainable_parameters()
    if resume_checkpoint:
        print("resuming from:", resume_checkpoint)

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    if args.smoke and repair_structure:
        update_audit = lora_b_update_audit(model)
        (output_dir / "weight_update_audit.json").write_text(
            json.dumps(update_audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(update_audit, ensure_ascii=True, indent=2))
        if update_audit["nonzero_lora_b_tensor_count"] == 0:
            raise RuntimeError("T1_v2 smoke failed: every LoRA B tensor is still zero")
        if not update_audit["structure_token_delta_nonzero"]:
            raise RuntimeError("T1_v2 smoke failed: structure token delta is still zero")
    eval_metrics = trainer.evaluate()
    trainer.save_metrics("eval", eval_metrics)

    final_adapter = output_dir / "final_adapter"
    trainer.save_model(str(final_adapter))
    tokenizer.save_pretrained(final_adapter)
    print("final adapter saved to:", final_adapter)

    if args.smoke or repair_structure:
        del trainer
        del model
        gc.collect()
        torch.cuda.empty_cache()
        reload_base = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=False,
            local_files_only=args.local_files_only,
            quantization_config=quantization_config,
            device_map={"": 0},
            dtype=compute_dtype,
            low_cpu_mem_usage=True,
        )
        reloaded = PeftModel.from_pretrained(reload_base, str(final_adapter))
        print("adapter reload: OK", type(reloaded).__name__)
        if repair_structure and not args.skip_generation_gate:
            gate_config = t1["generation_gate"]
            gate_rows = find_audit_rows(tokenized, list(gate_config["audit_rows"]))
            gate = run_free_generation_gate(
                reloaded,
                tokenizer,
                gate_rows,
                int(gate_config["max_new_tokens"]),
            )
            (output_dir / "generation_gate.json").write_text(
                json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(json.dumps(gate, ensure_ascii=True, indent=2))
            if not gate["passed"]:
                raise RuntimeError(
                    "T1_v2 free-generation gate failed; RLVR remains blocked"
                )


if __name__ == "__main__":
    main()
