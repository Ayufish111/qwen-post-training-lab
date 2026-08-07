"""Validate the frozen raw datasets and the optional tokenized cache."""

from __future__ import annotations

import json
import statistics
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.data_schema import validate_sft_record  # noqa: E402


PROJECT_CONFIG = yaml.safe_load(
    (PROJECT_DIR / "configs" / "project.yaml").read_text(encoding="utf-8")
)
MAX_LENGTH = PROJECT_CONFIG["model"]["max_seq_length"]

SFT_DIR = PROJECT_DIR / "data" / "sft"
CACHE_DIR = (
    PROJECT_DIR / "data" / "cache" / f"sft_qwen3_4b_{MAX_LENGTH}"
)
FILES = {
    "full": (SFT_DIR / "full_clean_10000.jsonl", 10_000),
    "clean": (SFT_DIR / "ablation_clean_2000.jsonl", 2_000),
    "raw": (SFT_DIR / "ablation_raw_2000.jsonl", 2_000),
}


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def question_key(row):
    text = row["messages"][0]["content"]
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(character for character in text if character.isalnum())


def stratum(row):
    metadata = row["metadata"]
    return metadata["task_bucket"], metadata["length_bucket"]


def validate_raw():
    failures = []
    datasets = {}

    for name, (path, expected_count) in FILES.items():
        if not path.exists():
            failures.append(f"missing file: {path}")
            continue
        rows = read_jsonl(path)
        datasets[name] = rows
        invalid = 0
        for index, row in enumerate(rows):
            errors = validate_sft_record(row)
            metadata = row.get("metadata")
            if not isinstance(metadata, dict):
                errors.append("metadata must be an object")
            else:
                for field in (
                    "id",
                    "reward",
                    "quality_score",
                    "quality_flags",
                    "task_bucket",
                    "length_bucket",
                ):
                    if field not in metadata:
                        errors.append(f"missing metadata.{field}")
            if errors:
                invalid += 1
                if invalid <= 3:
                    failures.append(f"{name}[{index}]: {'; '.join(errors)}")

        keys = [question_key(row) for row in rows]
        duplicates = len(keys) - len(set(keys))
        print(f"{name}: rows={len(rows)}, invalid={invalid}, duplicates={duplicates}")
        if len(rows) != expected_count:
            failures.append(f"{name}: expected {expected_count}, got {len(rows)}")
        if invalid:
            failures.append(f"{name}: {invalid} invalid records")
        if duplicates:
            failures.append(f"{name}: {duplicates} duplicate questions")

        if rows:
            scores = [row["metadata"]["quality_score"] for row in rows]
            rewards = [float(row["metadata"]["reward"]) for row in rows]
            print(
                f"  quality mean/median={statistics.mean(scores):.2f}/"
                f"{statistics.median(scores):.2f}; reward mean/median="
                f"{statistics.mean(rewards):.2f}/{statistics.median(rewards):.2f}"
            )

    if set(datasets) != set(FILES):
        return failures

    full_keys = {question_key(row) for row in datasets["full"]}
    clean_keys = {question_key(row) for row in datasets["clean"]}
    raw_keys = {question_key(row) for row in datasets["raw"]}
    clean_strata = Counter(stratum(row) for row in datasets["clean"])
    raw_strata = Counter(stratum(row) for row in datasets["raw"])

    print("clean subset of full:", clean_keys <= full_keys)
    print("raw/full overlap:", len(raw_keys & full_keys))
    print("raw/clean overlap:", len(raw_keys & clean_keys))
    print("clean/raw matched task-length strata:", clean_strata == raw_strata)

    if not clean_keys <= full_keys:
        failures.append("clean ablation is not a subset of full clean")
    if raw_keys & full_keys:
        failures.append("raw ablation overlaps full clean")
    if raw_keys & clean_keys:
        failures.append("raw and clean ablations overlap")
    if clean_strata != raw_strata:
        failures.append("clean/raw task-length strata differ")
    return failures


def validate_tokenized():
    if not CACHE_DIR.exists():
        print("tokenized cache: not present, skipped")
        return []

    from datasets import load_from_disk

    expected_splits = {
        "ablation_clean_train",
        "ablation_clean_validation",
        "ablation_raw_train",
        "ablation_raw_validation",
        "full_clean_train",
        "full_clean_validation",
    }
    dataset = load_from_disk(CACHE_DIR)
    failures = []
    if set(dataset) != expected_splits:
        failures.append(f"unexpected tokenized splits: {sorted(dataset)}")

    for split_name, split in dataset.items():
        invalid = 0
        for row in split:
            input_ids = row["input_ids"]
            attention_mask = row["attention_mask"]
            labels = row["labels"]
            if not (len(input_ids) == len(attention_mask) == len(labels)):
                invalid += 1
                continue
            if not input_ids or len(input_ids) > MAX_LENGTH:
                invalid += 1
                continue
            active = [i for i, label in enumerate(labels) if label != -100]
            if not active or any(labels[i] != input_ids[i] for i in active):
                invalid += 1
        print(f"tokenized {split_name}: rows={len(split)}, invalid={invalid}")
        if invalid:
            failures.append(f"tokenized {split_name}: {invalid} invalid rows")

    if len(dataset["ablation_clean_train"]) != len(dataset["ablation_raw_train"]):
        failures.append("paired train split sizes differ")
    if len(dataset["ablation_clean_validation"]) != len(
        dataset["ablation_raw_validation"]
    ):
        failures.append("paired validation split sizes differ")
    return failures


failures = validate_raw() + validate_tokenized()
if failures:
    print("\nVALIDATION FAILED")
    for failure in failures:
        print("-", failure)
    raise SystemExit(1)

print("\nVALIDATION PASSED")
