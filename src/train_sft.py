import argparse
from pathlib import Path

import torch
import yaml
from datasets import load_from_disk
from peft import (
    LoraConfig,
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
CONFIG_PATH = PROJECT_DIR / "configs" / "project.yaml"
PROJECT_CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

MODEL_NAME = PROJECT_CONFIG["model"]["name"]
MODEL_REVISION = PROJECT_CONFIG["model"]["revision"]
TRUST_REMOTE_CODE = PROJECT_CONFIG["model"]["trust_remote_code"]
MAX_LENGTH = PROJECT_CONFIG["model"]["max_seq_length"]
TOKENIZED_DATASET_DIR = (
    PROJECT_DIR / "data" / "cache" / f"sft_qwen3_4b_{MAX_LENGTH}"
)
OUTPUT_ROOT = PROJECT_DIR / PROJECT_CONFIG["sft"]["output_dir"]

TARGET_STRATEGIES = {
    "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "all-linear": "all-linear",
}

EXPERIMENTS = PROJECT_CONFIG["experiments"]


parser = argparse.ArgumentParser()
parser.add_argument(
    "--experiment",
    required=True,
    choices=EXPERIMENTS,
    help="S1/S2 compare LoRA targets, S3 tests raw data, S4 is final SFT.",
)
parser.add_argument(
    "--target-strategy",
    choices=TARGET_STRATEGIES,
    help="Required for S3/S4 after S1 and S2 select the winner.",
)
parser.add_argument(
    "--resume",
    action="store_true",
    help="Resume from the latest checkpoint in this experiment directory.",
)
args = parser.parse_args()

experiment = EXPERIMENTS[args.experiment].copy()
fixed_strategy = experiment["target_strategy"]

if fixed_strategy is None:
    if args.target_strategy is None:
        parser.error(
            f"{args.experiment} requires --target-strategy attention or all-linear"
        )
    experiment["target_strategy"] = args.target_strategy
elif args.target_strategy is not None and args.target_strategy != fixed_strategy:
    parser.error(
        f"{args.experiment} fixes --target-strategy to {fixed_strategy} "
        "to preserve the controlled experiment"
    )

target_modules = TARGET_STRATEGIES[experiment["target_strategy"]]
output_dir = OUTPUT_ROOT / args.experiment

last_checkpoint = (
    get_last_checkpoint(str(output_dir)) if output_dir.exists() else None
)
if args.resume and last_checkpoint is None:
    parser.error(f"No checkpoint found under {output_dir}")
if not args.resume and last_checkpoint is not None:
    raise RuntimeError(
        f"Existing checkpoint found: {last_checkpoint}. Use --resume to "
        "continue it, or move the old experiment directory before restarting."
    )
resume_checkpoint = last_checkpoint if args.resume else None

if PROJECT_CONFIG["data"]["current_sft_status"] != "approved":
    raise RuntimeError(
        "G1 data gate is not approved. Run validate_dataset.py, read "
        "reports/sft_data_audit.md, then set current_sft_status to approved."
    )

# Step 4: Load the tokenizer and the preprocessed datasets.
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    revision=MODEL_REVISION,
    trust_remote_code=TRUST_REMOTE_CODE,
)
tokenizer.padding_side = "right"

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenized_datasets = load_from_disk(TOKENIZED_DATASET_DIR)

required_splits = {
    experiment["train_split"],
    experiment["validation_split"],
}
missing_splits = required_splits - set(tokenized_datasets)
if missing_splits:
    raise RuntimeError(
        f"Tokenized dataset is missing splits: {sorted(missing_splits)}"
    )

train_dataset = tokenized_datasets[experiment["train_split"]]
validation_dataset = tokenized_datasets[experiment["validation_split"]]

# Step 5: Create the 4-bit model and attach the LoRA adapter.
if not torch.cuda.is_available():
    raise RuntimeError("QLoRA training requires an NVIDIA GPU.")

compute_dtype = (
    torch.bfloat16
    if torch.cuda.is_bf16_supported()
    else torch.float16
)

gpu_properties = torch.cuda.get_device_properties(0)
gpu_memory_gb = gpu_properties.total_memory / 1024**3
print(f"GPU preflight: {gpu_properties.name}, {gpu_memory_gb:.1f} GB")
print("compute dtype:", compute_dtype)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=PROJECT_CONFIG["qlora"]["load_in_4bit"],
    bnb_4bit_quant_type=PROJECT_CONFIG["qlora"]["quant_type"],
    bnb_4bit_use_double_quant=PROJECT_CONFIG["qlora"]["double_quant"],
    bnb_4bit_compute_dtype=compute_dtype,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    revision=MODEL_REVISION,
    trust_remote_code=TRUST_REMOTE_CODE,
    quantization_config=quantization_config,
    device_map={"": 0},
    dtype=compute_dtype,
    low_cpu_mem_usage=True,
)

model.config.use_cache = False
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=target_modules,
    r=PROJECT_CONFIG["qlora"]["r"],
    lora_alpha=PROJECT_CONFIG["qlora"]["lora_alpha"],
    lora_dropout=PROJECT_CONFIG["qlora"]["lora_dropout"],
    bias="none",
)

model = get_peft_model(model, lora_config)

# Step 6: Causal-LM evaluation uses the model's built-in eval_loss.
# A classification-style compute_metrics function is not needed here.

# Step 7: Configure the training arguments.
schedule_args = (
    {
        "eval_strategy": "steps",
        "eval_steps": 50,
        "save_strategy": "steps",
        "save_steps": 50,
    }
    if experiment["max_steps"] > 0
    else {
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
    }
)

training_args = TrainingArguments(
    output_dir=str(output_dir),
    num_train_epochs=experiment["num_train_epochs"],
    max_steps=experiment["max_steps"],
    per_device_train_batch_size=PROJECT_CONFIG["sft"]["per_device_train_batch_size"],
    per_device_eval_batch_size=PROJECT_CONFIG["sft"]["per_device_eval_batch_size"],
    gradient_accumulation_steps=PROJECT_CONFIG["sft"]["gradient_accumulation_steps"],
    learning_rate=PROJECT_CONFIG["sft"]["learning_rate"],
    lr_scheduler_type=PROJECT_CONFIG["sft"]["lr_scheduler_type"],
    warmup_ratio=PROJECT_CONFIG["sft"]["warmup_ratio"],
    optim="paged_adamw_8bit",
    bf16=compute_dtype == torch.bfloat16,
    fp16=compute_dtype == torch.float16,
    gradient_checkpointing=True,
    logging_steps=PROJECT_CONFIG["sft"]["logging_steps"],
    logging_first_step=True,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    seed=42,
    data_seed=42,
    run_name=args.experiment,
    **schedule_args,
)

# Step 8: Create the data collator and Trainer.
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    pad_to_multiple_of=8,
    label_pad_token_id=-100,
    return_tensors="pt",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,
)

print("experiment:", args.experiment)
print("train split:", experiment["train_split"])
print("validation split:", experiment["validation_split"])
print("target strategy:", experiment["target_strategy"])
print("output directory:", output_dir)
model.print_trainable_parameters()
print("train samples:", len(train_dataset))
print("validation samples:", len(validation_dataset))
print(
    "effective batch size:",
    PROJECT_CONFIG["sft"]["per_device_train_batch_size"]
    * PROJECT_CONFIG["sft"]["gradient_accumulation_steps"],
)
if experiment["max_steps"] > 0:
    print(
        "training budget:",
        experiment["max_steps"],
        "optimizer steps (overrides num_train_epochs)",
    )
if resume_checkpoint is not None:
    print("resuming from:", resume_checkpoint)

# Step 9: Train and evaluate the dataset.
train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
trainer.save_metrics("train", train_result.metrics)
trainer.save_state()

eval_metrics = trainer.evaluate()
trainer.save_metrics("eval", eval_metrics)

final_adapter_dir = output_dir / "final_adapter"
trainer.save_model(str(final_adapter_dir))
tokenizer.save_pretrained(final_adapter_dir)
print("final adapter saved to:", final_adapter_dir)
