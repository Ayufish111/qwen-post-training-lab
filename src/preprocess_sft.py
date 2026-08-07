"""Split and tokenize the three frozen SFT datasets."""

import json
import random
from pathlib import Path

import yaml
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROJECT_CONFIG = yaml.safe_load(
    (PROJECT_DIR / "configs" / "project.yaml").read_text(encoding="utf-8")
)

MODEL_NAME = PROJECT_CONFIG["model"]["name"]
MODEL_REVISION = PROJECT_CONFIG["model"]["revision"]
TRUST_REMOTE_CODE = PROJECT_CONFIG["model"]["trust_remote_code"]
MAX_LENGTH = PROJECT_CONFIG["model"]["max_seq_length"]
SEED = PROJECT_CONFIG["seed"]

SFT_DIR = PROJECT_DIR / "data" / "sft"
OUTPUT_DIR = (
    PROJECT_DIR / "data" / "cache" / f"sft_qwen3_4b_{MAX_LENGTH}"
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    revision=MODEL_REVISION,
    trust_remote_code=TRUST_REMOTE_CODE,
)


def read_jsonl(name):
    path = SFT_DIR / name
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def tokenize_example(example):
    messages = example["messages"]
    full = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
        return_dict=True,
    )
    prompt = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
    )

    input_ids = full["input_ids"]
    prompt_ids = prompt["input_ids"]
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Prompt tokens are not a prefix of the full dialogue")

    # Keep the assistant's <|im_end|> token in labels. It teaches Qwen where
    # the answer ends; only the prompt prefix is masked with -100.
    return {
        "input_ids": input_ids,
        "attention_mask": full["attention_mask"],
        "labels": [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :],
    }


def split_indexes(size):
    indexes = list(range(size))
    random.Random(SEED).shuffle(indexes)
    validation_size = max(1, round(size * 0.05))
    return indexes[validation_size:], indexes[:validation_size]


def tokenize_rows(rows):
    dataset = Dataset.from_list(rows)
    return dataset.map(tokenize_example, remove_columns=dataset.column_names)


full_rows = read_jsonl("full_clean_10000.jsonl")
clean_rows = read_jsonl("ablation_clean_2000.jsonl")
raw_rows = read_jsonl("ablation_raw_2000.jsonl")

# Clean and raw use the same shuffled pair indexes. If either side is too long,
# both are removed so E1/E2 and E3 keep equal sample counts.
pair_train_indexes, pair_validation_indexes = split_indexes(len(clean_rows))
clean_tokenized = tokenize_rows(clean_rows)
raw_tokenized = tokenize_rows(raw_rows)
valid_pair_indexes = [
    index
    for index in range(len(clean_rows))
    if len(clean_tokenized[index]["input_ids"]) <= MAX_LENGTH
    and len(raw_tokenized[index]["input_ids"]) <= MAX_LENGTH
]
valid_pair_indexes = set(valid_pair_indexes)
pair_train_indexes = [i for i in pair_train_indexes if i in valid_pair_indexes]
pair_validation_indexes = [i for i in pair_validation_indexes if i in valid_pair_indexes]

full_train_indexes, full_validation_indexes = split_indexes(len(full_rows))
full_tokenized = tokenize_rows(full_rows)
full_train_indexes = [
    i for i in full_train_indexes if len(full_tokenized[i]["input_ids"]) <= MAX_LENGTH
]
full_validation_indexes = [
    i for i in full_validation_indexes if len(full_tokenized[i]["input_ids"]) <= MAX_LENGTH
]

datasets = DatasetDict(
    {
        "ablation_clean_train": clean_tokenized.select(pair_train_indexes),
        "ablation_clean_validation": clean_tokenized.select(pair_validation_indexes),
        "ablation_raw_train": raw_tokenized.select(pair_train_indexes),
        "ablation_raw_validation": raw_tokenized.select(pair_validation_indexes),
        "full_clean_train": full_tokenized.select(full_train_indexes),
        "full_clean_validation": full_tokenized.select(full_validation_indexes),
    }
)

OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
datasets.save_to_disk(OUTPUT_DIR)

for name, split in datasets.items():
    print(name, len(split))
print("paired overlength rows removed:", len(clean_rows) - len(valid_pair_indexes))
print("saved:", OUTPUT_DIR)
