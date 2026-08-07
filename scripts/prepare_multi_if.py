"""Download and freeze the Chinese split of the official Multi-IF dataset."""

import csv
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

from datasets import load_dataset


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_DIR / "data" / "eval" / "multi_if_zh.csv"
SFT_DIR = PROJECT_DIR / "data" / "sft"
SOURCE_NAME = "facebook/Multi-IF"


def normalized(text):
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(character for character in text if character.isalnum())


def prompt_content(value):
    if value in (None, "", "None"):
        return None
    message = json.loads(value) if isinstance(value, str) else value
    return message["content"]


def sft_question_keys():
    keys = set()
    for filename in ("full_clean_10000.jsonl", "ablation_raw_2000.jsonl"):
        with (SFT_DIR / filename).open("r", encoding="utf-8") as file:
            for line in file:
                row = json.loads(line)
                keys.add(normalized(row["messages"][0]["content"]))
    return keys


dataset = load_dataset(SOURCE_NAME, split="train")
language_counts = Counter(dataset["language"])
rows = [row for row in dataset if row["language"] == "Chinese"]

if not rows:
    raise RuntimeError(
        f"No Chinese rows found. Available languages: {dict(language_counts)}"
    )

# Hash order freezes a deterministic pilot prefix without depending on source order.
rows.sort(
    key=lambda row: hashlib.sha256(str(row["key"]).encode("utf-8")).hexdigest()
)

training_questions = sft_question_keys()
overlaps = []
for row in rows:
    for turn in (1, 2, 3):
        content = prompt_content(row.get(f"turn_{turn}_prompt"))
        if content and normalized(content) in training_questions:
            overlaps.append((row["key"], turn))

if overlaps:
    raise RuntimeError(
        f"Found {len(overlaps)} exact prompt overlaps with SFT data: {overlaps[:10]}"
    )

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fieldnames = list(dataset.column_names)
with OUTPUT_PATH.open("w", encoding="utf-8-sig", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

digest = hashlib.sha256(OUTPUT_PATH.read_bytes()).hexdigest()
print("source:", SOURCE_NAME)
print("all language counts:", dict(sorted(language_counts.items())))
print("Chinese rows:", len(rows))
print("exact SFT prompt overlaps:", len(overlaps))
print("SHA256:", digest)
print("saved:", OUTPUT_PATH)
