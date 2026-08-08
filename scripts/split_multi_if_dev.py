"""Freeze a deterministic development/test split for the Multi-IF CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Iterable


DEFAULT_INPUT = Path("data/eval/multi_if_zh.csv")
DEFAULT_OUTPUT_DIR = Path("data/eval")
DEFAULT_MANIFEST = Path("reports/manifests/multi_if_split_manifest.json")
EXPECTED_ROWS = 454
EXPECTED_DEV_SIZE = 80
REQUIRED_COLUMNS = {
    "key",
    "language",
    "turn_1_prompt",
    "turn_1_instruction_id_list",
    "turn_1_kwargs",
    "turn_2_prompt",
    "turn_2_instruction_id_list",
    "turn_2_kwargs",
    "turn_3_prompt",
    "turn_3_instruction_id_list",
    "turn_3_kwargs",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_prompt(value: str) -> str:
    """Normalize Unicode and whitespace for a leakage diagnostic only."""

    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def prompt_set(row: dict[str, str]) -> set[str]:
    prompts: set[str] = set()
    for turn in (1, 2, 3):
        field = f"turn_{turn}_prompt"
        raw = row.get(field, "")
        if not raw:
            continue
        try:
            message = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {field} for row {row.get('key')}") from exc
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ValueError(f"{field} for row {row.get('key')} is not a user message")
        prompt = normalized_prompt(message["content"])
        if prompt:
            prompts.add(prompt)
    return prompts


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        missing = REQUIRED_COLUMNS - set(fieldnames)
        if missing:
            raise ValueError(f"Input is missing required columns: {sorted(missing)}")
        rows = list(reader)

    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, found {len(rows)}")
    keys = [row["key"] for row in rows]
    if any(not key for key in keys):
        raise ValueError("Input contains an empty key")
    if len(set(keys)) != len(keys):
        raise ValueError("Input contains duplicate keys")
    for row in rows:
        prompt_set(row)
    return fieldnames, rows


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_ids(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{row['key']}\n" for row in rows), encoding="utf-8")


def split_rows(rows: list[dict[str, str]], seed: int, dev_size: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    ordered = sorted(rows, key=lambda row: row["key"])
    random.Random(seed).shuffle(ordered)
    return ordered[:dev_size], ordered[dev_size:]


def split_metadata(path: Path, ids_path: Path, rows: list[dict[str, str]]) -> dict[str, object]:
    ids = [row["key"] for row in rows]
    return {
        "path": path.as_posix(),
        "ids_path": ids_path.as_posix(),
        "rows": len(rows),
        "unique_ids": len(set(ids)),
        "duplicate_ids": len(ids) - len(set(ids)),
        "sha256": sha256_file(path),
        "ids_sha256": sha256_file(ids_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-size", type=int, default=EXPECTED_DEV_SIZE)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow replacing an existing split and manifest after verification.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dev_size <= 0 or args.dev_size >= EXPECTED_ROWS:
        raise ValueError("--dev-size must be between 1 and 453")

    output_paths = [
        args.output_dir / "multi_if_dev.csv",
        args.output_dir / "multi_if_test.csv",
        args.output_dir / "multi_if_dev_ids.txt",
        args.output_dir / "multi_if_test_ids.txt",
        args.manifest,
    ]
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.force:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing split outputs: {names}; use --force")

    fieldnames, rows = read_rows(args.input)
    dev_rows, test_rows = split_rows(rows, args.seed, args.dev_size)
    dev_ids = {row["key"] for row in dev_rows}
    test_ids = {row["key"] for row in test_rows}
    id_intersection = dev_ids & test_ids
    if id_intersection or len(dev_rows) + len(test_rows) != len(rows):
        raise AssertionError("Split does not partition the input rows")

    dev_prompts = {prompt for row in dev_rows for prompt in prompt_set(row)}
    test_prompts = {prompt for row in test_rows for prompt in prompt_set(row)}
    prompt_overlap = dev_prompts & test_prompts
    prompt_overlap_bytes = "\n".join(sorted(prompt_overlap)).encode("utf-8")

    dev_csv = args.output_dir / "multi_if_dev.csv"
    test_csv = args.output_dir / "multi_if_test.csv"
    dev_ids_path = args.output_dir / "multi_if_dev_ids.txt"
    test_ids_path = args.output_dir / "multi_if_test_ids.txt"
    write_csv(dev_csv, fieldnames, dev_rows)
    write_csv(test_csv, fieldnames, test_rows)
    write_ids(dev_ids_path, dev_rows)
    write_ids(test_ids_path, test_rows)

    manifest = {
        "schema_version": "multi_if_split/v1",
        "script_version": "2026-08-08",
        "input": {
            "path": args.input.as_posix(),
            "rows": len(rows),
            "unique_ids": len({row["key"] for row in rows}),
            "sha256": sha256_file(args.input),
        },
        "algorithm": {
            "sort_key": "key ascending",
            "shuffle": "random.Random(seed).shuffle",
            "seed": args.seed,
            "dev_size": len(dev_rows),
            "test_size": len(test_rows),
        },
        "splits": {
            "dev": split_metadata(dev_csv, dev_ids_path, dev_rows),
            "test": split_metadata(test_csv, test_ids_path, test_rows),
        },
        "checks": {
            "input_duplicate_ids": len(rows) - len({row["key"] for row in rows}),
            "dev_duplicate_ids": len(dev_rows) - len(dev_ids),
            "test_duplicate_ids": len(test_rows) - len(test_ids),
            "dev_test_id_intersection_count": len(id_intersection),
            "normalized_prompt_overlap_count": len(prompt_overlap),
            "normalized_prompt_overlap_sha256": sha256_bytes(prompt_overlap_bytes),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
