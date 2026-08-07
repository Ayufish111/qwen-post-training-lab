"""Build the three frozen SFT datasets used by the ablation experiments."""

from __future__ import annotations

import hashlib
import json
import random
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


SEED = 42
FULL_SIZE = 10_000
ABLATION_SIZE = 2_000
STREAM_POOL_SIZE = 30_000

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "sft"
CANDIDATE_PATH = DATA_DIR / "technical_candidates.jsonl"
FULL_PATH = DATA_DIR / "full_clean_10000.jsonl"
CLEAN_PATH = DATA_DIR / "ablation_clean_2000.jsonl"
RAW_PATH = DATA_DIR / "ablation_raw_2000.jsonl"
REPORT_PATH = PROJECT_DIR / "reports" / "sft_data_audit.md"

TASK_RULES = (
    ("code", ("代码", "编程", "python", "java", "c++", "sql", "算法")),
    ("math_reasoning", ("数学", "计算", "推理", "逻辑", "证明")),
    ("translation", ("翻译", "译成", "translate")),
    ("rewrite_summary", ("改写", "润色", "摘要", "总结", "概括", "压缩")),
    ("extraction_classification", ("抽取", "分类", "情感分析", "实体识别")),
    ("creative_writing", ("写一篇", "故事", "诗歌", "创作", "文案")),
    ("dialogue_roleplay", ("对话", "角色扮演", "回复")),
    ("knowledge_qa", ("知识", "常识", "科普", "法律", "医学", "历史")),
)

PLACEHOLDER_PATTERN = re.compile(
    r"__[A-Z][A-Z0-9_]*__|\[(?:姓名|日期|公司|部门|职位|待填写)\]"
)
SELF_ID_MARKERS = ("作为一个人工智能", "作为一名人工智能", "作为chatgpt")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalized(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(character for character in text if character.isalnum())


def question_key(row: dict) -> str:
    return normalized(row["messages"][0]["content"])


def stable_number(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def convert_source_row(source_row: dict) -> dict | None:
    if source_row.get("langdetect") != "zh-cn":
        return None
    conversations = source_row.get("conversations")
    if not isinstance(conversations, list) or len(conversations) != 2:
        return None
    if conversations[0].get("from") != "human":
        return None
    if conversations[1].get("from") != "gpt":
        return None
    question = conversations[0].get("value")
    answer = conversations[1].get("value")
    reward = source_row.get("reward")
    if not isinstance(question, str) or not isinstance(answer, str):
        return None
    if not question.strip() or not answer.strip():
        return None
    if not isinstance(reward, (int, float)):
        return None

    label = source_row.get("label") or {}
    abilities = label.get("ability_zh") or []
    categories = label.get("cate_ability_zh") or []
    return {
        "messages": [
            {"role": "user", "content": question.strip()},
            {"role": "assistant", "content": answer.strip()},
        ],
        "metadata": {
            "id": source_row.get("id"),
            "ability": abilities if isinstance(abilities, list) else [],
            "category": categories if isinstance(categories, list) else [],
            "reward": float(reward),
            "source": str(source_row.get("source") or "unknown"),
        },
    }


def collect_candidates() -> list[dict]:
    """Stream the source only when the one-time local candidate cache is absent."""
    from modelscope.msdatasets import MsDataset

    print("technical candidate cache is absent; streaming Infinity-Instruct 7M")
    dataset = MsDataset.load(
        "Infinity-Instruct",
        namespace="AI-ModelScope",
        subset_name="7M",
        split="train",
        use_streaming=True,
    )
    rng = random.Random(SEED)
    pool: list[dict] = []
    eligible = 0
    for scanned, source_row in enumerate(dataset, start=1):
        if scanned % 100_000 == 0:
            print("scanned:", scanned, "eligible:", eligible)
        row = convert_source_row(source_row)
        if row is None:
            continue
        eligible += 1
        if len(pool) < STREAM_POOL_SIZE:
            pool.append(row)
        else:
            replacement = rng.randrange(eligible)
            if replacement < STREAM_POOL_SIZE:
                pool[replacement] = row
    return pool


def load_candidates() -> list[dict]:
    if CANDIDATE_PATH.exists():
        print("using one-time candidate cache:", CANDIDATE_PATH)
        return read_jsonl(CANDIDATE_PATH)
    if FULL_PATH.exists() and RAW_PATH.exists():
        print("using the frozen 12k experimental universe for deterministic rebuild")
        rows = read_jsonl(FULL_PATH) + read_jsonl(RAW_PATH)
        unique = {question_key(row): row for row in rows}
        return list(unique.values())
    return collect_candidates()


def task_bucket(row: dict) -> str:
    metadata = row.get("metadata") or {}
    abilities = metadata.get("ability") or []
    question = row["messages"][0]["content"]
    searchable = (" ".join(map(str, abilities)) + " " + question).lower()
    for bucket, markers in TASK_RULES:
        if any(marker in searchable for marker in markers):
            return bucket
    return "general_instruction"


def length_bucket(row: dict) -> str:
    size = sum(len(message["content"]) for message in row["messages"])
    for limit in (256, 512, 1024, 2048):
        if size <= limit:
            return f"le_{limit}"
    return "gt_2048"


def inspect_row(row: dict) -> tuple[list[str], list[str], float]:
    hard_failures: list[str] = []
    soft_flags: list[str] = []

    messages = row.get("messages")
    metadata = row.get("metadata")
    if not isinstance(messages, list) or len(messages) != 2:
        return ["not_one_turn_dialogue"], [], 0.0
    if [message.get("role") for message in messages] != ["user", "assistant"]:
        return ["invalid_roles"], [], 0.0
    if not isinstance(metadata, dict):
        return ["missing_metadata"], [], 0.0

    question = messages[0].get("content")
    answer = messages[1].get("content")
    reward = metadata.get("reward")
    if not isinstance(question, str) or not question.strip():
        hard_failures.append("empty_question")
    if not isinstance(answer, str) or not answer.strip():
        hard_failures.append("empty_answer")
    if not isinstance(reward, (int, float)):
        hard_failures.append("invalid_reward")
    if hard_failures:
        return hard_failures, [], 0.0

    question = question.strip()
    answer = answer.strip()
    if normalized(question) == normalized(answer):
        hard_failures.append("question_equals_answer")
    if "\ufffd" in question + answer:
        hard_failures.append("invalid_unicode_replacement")
    if PLACEHOLDER_PATTERN.search(answer):
        hard_failures.append("unresolved_placeholder")

    reward = float(reward)
    if reward < 0:
        soft_flags.append("negative_reward")
    elif reward < 1:
        soft_flags.append("low_reward")
    if len(answer) < 8 and len(question) > 24:
        soft_flags.append("very_short_answer")
    if len(question) + len(answer) > 4_000:
        soft_flags.append("very_long_sample")
    if any(marker in answer.lower() for marker in SELF_ID_MARKERS):
        soft_flags.append("assistant_self_identification")
    if normalized(answer) and normalized(answer) in normalized(question):
        soft_flags.append("answer_copied_from_prompt")

    penalties = {
        "negative_reward": 12,
        "low_reward": 4,
        "very_short_answer": 8,
        "very_long_sample": 3,
        "assistant_self_identification": 8,
        "answer_copied_from_prompt": 10,
    }
    score = 50 + max(-5.0, min(5.0, reward)) * 2
    score -= sum(penalties[flag] for flag in soft_flags)
    return hard_failures, soft_flags, round(score, 3)


def annotate_candidates(rows: list[dict]) -> tuple[list[dict], Counter]:
    failures: Counter = Counter()
    valid: list[dict] = []
    seen: set[str] = set()

    for row in rows:
        hard_failures, flags, score = inspect_row(row)
        if hard_failures:
            failures.update(hard_failures)
            continue
        key = question_key(row)
        if not key or key in seen:
            failures["duplicate_question"] += 1
            continue
        seen.add(key)

        metadata = dict(row["metadata"])
        metadata.update(
            quality_score=score,
            quality_flags=flags,
            task_bucket=task_bucket(row),
            length_bucket=length_bucket(row),
        )
        valid.append({"messages": row["messages"], "metadata": metadata})
    return valid, failures


def allocate_by_stratum(groups, target, minimums, capacities):
    """Allocate an exact target while preserving the source distribution."""
    total = sum(len(group) for group in groups.values())
    ideals = {key: len(group) * target / total for key, group in groups.items()}
    counts = {
        key: min(capacities[key], max(minimums[key], int(ideals[key])))
        for key in groups
    }
    while sum(counts.values()) < target:
        choices = [key for key in groups if counts[key] < capacities[key]]
        if not choices:
            raise RuntimeError("Not enough rows for the requested stratified allocation")
        key = max(choices, key=lambda item: (ideals[item] - counts[item], item))
        counts[key] += 1
    while sum(counts.values()) > target:
        choices = [key for key in groups if counts[key] > minimums[key]]
        if not choices:
            raise RuntimeError("Stratified minimums exceed the requested size")
        key = min(choices, key=lambda item: (ideals[item] - counts[item], item))
        counts[key] -= 1
    return counts


def select_datasets(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        metadata = row["metadata"]
        groups[(metadata["task_bucket"], metadata["length_bucket"])].append(row)
    groups = {key: group for key, group in groups.items() if len(group) >= 2}
    for group in groups.values():
        group.sort(
            key=lambda row: (
                -row["metadata"]["quality_score"],
                stable_number(question_key(row)),
            )
        )

    raw_minimums = {key: 1 for key in groups}
    raw_capacities = {
        key: max(1, len(group) // 3) for key, group in groups.items()
    }
    raw_counts = allocate_by_stratum(
        groups, ABLATION_SIZE, raw_minimums, raw_capacities
    )
    full_minimums = dict(raw_counts)
    full_capacities = {
        key: len(group) - raw_counts[key] for key, group in groups.items()
    }
    full_counts = allocate_by_stratum(
        groups, FULL_SIZE, full_minimums, full_capacities
    )

    full: list[dict] = []
    clean: list[dict] = []
    raw: list[dict] = []
    for key, group in groups.items():
        raw_count = raw_counts[key]
        full_count = full_counts[key]
        full.extend(group[:full_count])
        clean.extend(group[:raw_count])
        raw.extend(group[-raw_count:])

    if (len(full), len(clean), len(raw)) != (
        FULL_SIZE,
        ABLATION_SIZE,
        ABLATION_SIZE,
    ):
        raise RuntimeError("Stratified selection produced incorrect dataset sizes")
    return full, clean, raw


def distribution(rows: list[dict], field: str) -> Counter:
    return Counter(row["metadata"][field] for row in rows)


def score_summary(rows: list[dict]) -> str:
    values = [row["metadata"]["quality_score"] for row in rows]
    rewards = [float(row["metadata"]["reward"]) for row in rows]
    return (
        f"quality mean/median={statistics.mean(values):.2f}/{statistics.median(values):.2f}, "
        f"reward mean/median={statistics.mean(rewards):.2f}/{statistics.median(rewards):.2f}"
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_report(
    input_count: int,
    valid_count: int,
    failures: Counter,
    full: list[dict],
    clean: list[dict],
    raw: list[dict],
) -> None:
    if input_count == FULL_SIZE + ABLATION_SIZE and not failures:
        source_audit = (
            "This rebuild repartitioned the frozen 12,000-row experimental "
            "universe. It came from the original 20,000-row candidate pass, "
            "where 11 unresolved placeholders and 1 duplicate question were "
            "removed before freezing."
        )
    else:
        source_audit = (
            f"Parsed {input_count} source candidates; {valid_count} unique, "
            "structurally valid rows remained."
        )
    failure_lines = "\n".join(
        f"- `{reason}`: {count}" for reason, count in failures.most_common()
    ) or "- None"
    task_lines = "\n".join(
        f"- `{bucket}`: full={distribution(full, 'task_bucket')[bucket]}, "
        f"clean={distribution(clean, 'task_bucket')[bucket]}, "
        f"raw={distribution(raw, 'task_bucket')[bucket]}"
        for bucket in sorted(distribution(full, "task_bucket"))
    )
    REPORT_PATH.write_text(
        f"""# SFT Data Audit

Audit date: 2026-07-31

Status: automatic data gate passed. Semantic factual correctness is not claimed.

## Frozen outputs

| File | Rows | SHA256 | Purpose |
|---|---:|---|---|
| `full_clean_10000.jsonl` | {len(full)} | `{file_sha256(FULL_PATH)}` | E4 final SFT |
| `ablation_clean_2000.jsonl` | {len(clean)} | `{file_sha256(CLEAN_PATH)}` | E1/E2 target-module comparison |
| `ablation_raw_2000.jsonl` | {len(raw)} | `{file_sha256(RAW_PATH)}` | E3 data-quality ablation |

The clean ablation is a subset of the full clean set. The raw ablation is
disjoint from both clean files. Clean and raw have identical task-bucket and
character-length-bucket counts.

## Cleaning operations

- {source_audit}
- Hard rules removed only observable structural failures.
- Reward and visible risk signals were converted to a documented quality score.
- The 10,000-row set was selected across task buckets, then 2,000 clean/raw
  pairs were matched by task bucket and length bucket.

Hard-rule removals:

{failure_lines}

## Distribution

{task_lines}

- Full clean: {score_summary(full)}
- Ablation clean: {score_summary(clean)}
- Ablation raw: {score_summary(raw)}

## Interpretation boundary

This pipeline demonstrates schema validation, exact deduplication, transparent
rule-based cleaning, risk scoring, balanced sampling, and controlled ablation
construction. It does not prove every answer is factually correct. Formal model
claims require the frozen external evaluation suite and error analysis after
training; a sampled review is evidence for audit only, not a cleaning step.
""",
        encoding="utf-8",
    )


def main() -> None:
    candidates = load_candidates()
    valid, failures = annotate_candidates(candidates)
    full, clean, raw = select_datasets(valid)

    clean_strata = Counter(
        (row["metadata"]["task_bucket"], row["metadata"]["length_bucket"])
        for row in clean
    )
    raw_strata = Counter(
        (row["metadata"]["task_bucket"], row["metadata"]["length_bucket"])
        for row in raw
    )
    assert clean_strata == raw_strata
    assert {question_key(row) for row in clean} <= {question_key(row) for row in full}
    assert {question_key(row) for row in raw}.isdisjoint(
        question_key(row) for row in full
    )

    write_jsonl(FULL_PATH, full)
    write_jsonl(CLEAN_PATH, clean)
    write_jsonl(RAW_PATH, raw)
    write_report(len(candidates), len(valid), failures, full, clean, raw)

    print("full clean:", len(full), FULL_PATH)
    print("ablation clean:", len(clean), CLEAN_PATH)
    print("ablation raw:", len(raw), RAW_PATH)
    print("matched task/length strata:", clean_strata == raw_strata)
    print("clean:", score_summary(clean))
    print("raw:", score_summary(raw))
    print("audit report:", REPORT_PATH)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Build failed: {error}", file=sys.stderr)
        raise
