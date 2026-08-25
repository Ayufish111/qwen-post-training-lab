"""Build the frozen 50-prompt Chinese quality guardrail.

The guardrail is not a benchmark and has no reference answers. It detects
obvious helpfulness/readability regressions alongside the programmatic
constraint benchmark. Selection excludes every RLVR source id and performs
exact normalized-prompt overlap checks against the frozen Multi-IF boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_DIR / "data" / "sft" / "full_clean_10000.jsonl"
RLVR_PATHS = (
    PROJECT_DIR / "data" / "rlvr" / "constraint_train_2000.jsonl",
    PROJECT_DIR / "data" / "rlvr" / "constraint_validation_100.jsonl",
)
MULTI_IF_PATHS = (
    PROJECT_DIR / "data" / "eval" / "multi_if_dev.csv",
    PROJECT_DIR / "data" / "eval" / "multi_if_test.csv",
)
OUTPUT_PATH = PROJECT_DIR / "data" / "eval" / "chinese_quality_guardrail_50.jsonl"
MANIFEST_PATH = (
    PROJECT_DIR / "reports" / "manifests" / "chinese_quality_guardrail_manifest.json"
)
SEED = 42
MIN_PROMPT_CHARACTERS = 20
MAX_PROMPT_CHARACTERS = 1500
MIN_QUALITY_SCORE = 52.0
QUOTAS = {
    "code": 5,
    "creative_writing": 6,
    "dialogue_roleplay": 5,
    "extraction_classification": 5,
    "general_instruction": 7,
    "knowledge_qa": 7,
    "math_reasoning": 6,
    "rewrite_summary": 5,
    "translation": 4,
}
MANUAL_EXCLUSIONS = {
    "7163655": "source sentence is already in the requested progressive form",
    "3347798": "translation target language is not specified",
    "3427272": "claims Dutch input but provides Chinese text",
    "5268267": "asks to replace a verb without providing the source sentence",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_prompt(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower()
    return " ".join(value.split())


def stable_key(source_id: str, prompt: str, seed: int) -> str:
    value = f"{seed}:{source_id}:{normalize_prompt(prompt)}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_rlvr_source_ids(paths: Sequence[Path]) -> set[str]:
    return {
        str(row["metadata"]["source_id"])
        for path in paths
        for row in read_jsonl(path)
    }


def load_multi_if_prompt_keys(paths: Sequence[Path]) -> set[str]:
    # Reuse the already-tested parser without displaying untouched test text.
    sys.path.insert(0, str(PROJECT_DIR))
    from scripts.build_constraint_rlvr_data import load_multi_if_prompts

    return load_multi_if_prompts(list(paths))


def select_rows(
    source_rows: Sequence[dict[str, Any]],
    *,
    excluded_source_ids: set[str],
    excluded_prompt_keys: set[str],
    quotas: dict[str, int],
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_prompts: set[str] = set()
    for row in source_rows:
        metadata = row.get("metadata") or {}
        source_id = str(metadata.get("id", ""))
        bucket = metadata.get("task_bucket")
        messages = row.get("messages") or []
        if (
            bucket not in quotas
            or source_id in excluded_source_ids
            or source_id in MANUAL_EXCLUSIONS
        ):
            continue
        if len(messages) != 2 or messages[0].get("role") != "user":
            continue
        prompt = messages[0].get("content")
        if not isinstance(prompt, str):
            continue
        prompt = prompt.strip()
        prompt_key = normalize_prompt(prompt)
        if (
            not MIN_PROMPT_CHARACTERS <= len(prompt) <= MAX_PROMPT_CHARACTERS
            or not prompt_key
            or prompt_key in seen_prompts
            or prompt_key in excluded_prompt_keys
            or "\ufffd" in prompt
            or float(metadata.get("quality_score", 0.0)) < MIN_QUALITY_SCORE
        ):
            continue
        seen_prompts.add(prompt_key)
        grouped[bucket].append(row)

    selected: list[dict[str, Any]] = []
    for bucket, required in sorted(quotas.items()):
        candidates = sorted(
            grouped[bucket],
            key=lambda row: stable_key(
                str(row["metadata"]["id"]),
                row["messages"][0]["content"],
                seed,
            ),
        )
        if len(candidates) < required:
            raise RuntimeError(
                f"Guardrail bucket {bucket!r} has {len(candidates)} candidates; "
                f"{required} required"
            )
        for row in candidates[:required]:
            metadata = row["metadata"]
            selected.append(
                {
                    "id": "pending",
                    "messages": [
                        {
                            "role": "user",
                            "content": row["messages"][0]["content"].strip(),
                        }
                    ],
                    "category": bucket,
                    "metadata": {
                        "source_id": str(metadata["id"]),
                        "quality_score": float(metadata["quality_score"]),
                        "source": metadata.get("source", "unknown"),
                    },
                }
            )

    selected.sort(
        key=lambda row: stable_key(
            row["metadata"]["source_id"],
            row["messages"][0]["content"],
            seed,
        )
    )
    for index, row in enumerate(selected, start=1):
        row["id"] = f"quality-zh-{index:03d}"
    return selected


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for path in (args.output, args.manifest):
        if path.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite frozen output: {path}")
    for path in (args.source, *RLVR_PATHS, *MULTI_IF_PATHS):
        if not path.exists():
            raise FileNotFoundError(path)

    rlvr_source_ids = load_rlvr_source_ids(RLVR_PATHS)
    multi_if_prompt_keys = load_multi_if_prompt_keys(MULTI_IF_PATHS)
    selected = select_rows(
        read_jsonl(args.source),
        excluded_source_ids=rlvr_source_ids,
        excluded_prompt_keys=multi_if_prompt_keys,
        quotas=QUOTAS,
        seed=args.seed,
    )
    expected_size = sum(QUOTAS.values())
    if expected_size != 50 or len(selected) != expected_size:
        raise AssertionError("Guardrail size is not exactly 50")

    write_jsonl(args.output, selected)
    selected_source_ids = {row["metadata"]["source_id"] for row in selected}
    selected_prompt_keys = {
        normalize_prompt(row["messages"][0]["content"]) for row in selected
    }
    manifest = {
        "schema_version": "chinese_quality_guardrail/v1",
        "status": "frozen_before_r0",
        "purpose": "readability_and_helpfulness_regression_guardrail_not_a_benchmark",
        "seed": args.seed,
        "selection": {
            "source_task_bucket_quotas": QUOTAS,
            "quota_basis": "upstream_heuristic_labels_not_manual_ground_truth",
            "minimum_quality_score": MIN_QUALITY_SCORE,
            "prompt_character_range": [MIN_PROMPT_CHARACTERS, MAX_PROMPT_CHARACTERS],
            "reference_answers_included": False,
            "manual_prompt_exclusions": MANUAL_EXCLUSIONS,
        },
        "source": {
            "path": args.source.as_posix(),
            "sha256": sha256_file(args.source),
        },
        "output": {
            "path": args.output.as_posix(),
            "rows": len(selected),
            "sha256": sha256_file(args.output),
        },
        "script": {
            "path": Path(__file__).resolve().as_posix(),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "counts": {
            "categories": dict(sorted(Counter(row["category"] for row in selected).items())),
            "unique_ids": len({row["id"] for row in selected}),
            "unique_source_ids": len(selected_source_ids),
            "unique_prompts": len(selected_prompt_keys),
        },
        "checks": {
            "rlvr_source_id_overlap": len(selected_source_ids & rlvr_source_ids),
            "multi_if_normalized_prompt_overlap": len(selected_prompt_keys & multi_if_prompt_keys),
        },
        "evaluation_policy": {
            "does_not_affect_primary_score": True,
            "not_used_for_tuning_or_p2_decision": True,
            "outputs_reviewed_only_after_all_compared_models_finish": True,
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
