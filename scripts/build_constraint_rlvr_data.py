"""Build the frozen, independently sourced RLVR constraint prompt dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "rlvr.yaml"
GENERIC_TERMS = {
    "一个",
    "一些",
    "以下",
    "以及",
    "什么",
    "内容",
    "可以",
    "回答",
    "如何",
    "应该",
    "提供",
    "描述",
    "相关",
    "进行",
    "这个",
    "通过",
    "问题",
    "需要",
    "为什么",
}

REQUEST_PREFIX_RE = re.compile(
    r"^(?:请求|请问|请你|请|你能否|你能|能否|帮我|给我|我想要|我想|我们需要|我们想要)"
)
ACTION_PREFIX_RE = re.compile(
    r"^(?:(?:把)?这(?:句话|段话|段落)?翻译成|撰写|编写|拼写|写下|写出|写|生成|创建|"
    r"设计|列出|解释|说明|描述|提供|输出|讨论|探讨|分析|比较|总结|翻译|改写|回答|"
    r"完成|制作|实现|开发|阐明|鉴赏|赞扬|赞许|归纳|推荐|指导|指点|找出|找到|研究|"
    r"基于|给出|扩展|解析|建议)"
)
EXPLICIT_OUTPUT_CONSTRAINT_RE = re.compile(
    r"(?:(?:最多|不超过|不得超过|少于|至少|不少于|多于|恰好|正好|限制在)"
    r".{0,10}(?:个)?(?:字|词|句话|句|段落|段)|"
    r"[\d一二三四五六七八九十]+(?:到|至|-)[\d一二三四五六七八九十]+(?:句话|句|段落|段)|"
    r"(?:单行|一句话)(?:简介|回答|描述|总结|摘要))"
)
DETERMINER_PREFIX_RE = re.compile(
    r"^(?:一篇|一段|一个|一种|一份|一些|几个|几条|五个|以下|有关|其(?!他))"
)
TOPIC_PREFIX_RE = re.compile(
    r"^(?:关于|针对|围绕|例如|比如|如(?!何|此)|包括|采用|使用|利用)"
)
TOPIC_MARKER_RE = re.compile(
    r"(?:关于|针对|围绕|例如|比如|包括|采用|使用|利用)"
    r"([\u3400-\u9fffA-Za-z0-9_+#.·\- ]{2,24}?)"
    r"(?=的(?:文章|内容|方法|脚本|代码|方案|问题)|[，。！？；、,!?;\n]|$)"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_prompt(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower()
    return re.sub(r"\s+", " ", value).strip()


def has_explicit_output_constraint(value: str) -> bool:
    return bool(EXPLICIT_OUTPUT_CONSTRAINT_RE.search(value))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def allocation(total: int, ratios: dict[Any, float]) -> dict[Any, int]:
    exact = {key: total * value for key, value in ratios.items()}
    counts = {key: math.floor(value) for key, value in exact.items()}
    remainder = total - sum(counts.values())
    order = sorted(exact, key=lambda key: (-(exact[key] - counts[key]), str(key)))
    for key in order[:remainder]:
        counts[key] += 1
    return counts


def expanded_schedule(counts: dict[Any, int], rng: random.Random) -> list[Any]:
    values = [key for key, count in counts.items() for _ in range(count)]
    rng.shuffle(values)
    return values


def load_multi_if_prompts(paths: list[Path]) -> set[str]:
    prompts: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                for turn in (1, 2, 3):
                    raw = row.get(f"turn_{turn}_prompt", "")
                    if not raw:
                        continue
                    message = json.loads(raw)
                    prompts.add(normalize_prompt(message["content"]))
    return prompts


def select_sources(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    multi_if_prompts: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    data = config["data"]
    total = data["train_size"] + data["validation_size"]
    quotas = allocation(total, data["task_bucket_ratios"])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    overlap_removed = 0
    intrinsic_constraint_removed = 0

    for row in rows:
        metadata = row["metadata"]
        bucket = metadata["task_bucket"]
        if bucket not in quotas:
            continue
        if float(metadata["quality_score"]) < data["minimum_quality_score"]:
            continue
        source_prompt = normalize_prompt(row["messages"][0]["content"])
        if source_prompt in multi_if_prompts:
            overlap_removed += 1
            continue
        if has_explicit_output_constraint(row["messages"][0]["content"]):
            intrinsic_constraint_removed += 1
            continue
        grouped[bucket].append(row)

    selected: list[dict[str, Any]] = []
    for bucket, required in quotas.items():
        candidates = sorted(
            grouped[bucket],
            key=lambda row: stable_key(
                f"{config['seed']}:{row['metadata']['id']}:"
                f"{row['messages'][0]['content']}"
            ),
        )
        if len(candidates) < required:
            raise ValueError(
                f"Not enough {bucket} candidates: {len(candidates)} < {required}"
            )
        selected.extend(candidates[:required])

    return selected, {
        "source_prompt_overlap_removed": overlap_removed,
        "source_intrinsic_constraint_removed": intrinsic_constraint_removed,
        **quotas,
    }


def clean_prompt_term(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"[`*_#<>\[\]{}()（）]", "", value).strip()
    value = re.sub(r"^[\d.、\s]+", "", value)
    previous = None
    while value and value != previous:
        previous = value
        value = REQUEST_PREFIX_RE.sub("", value).strip()
        value = ACTION_PREFIX_RE.sub("", value).strip()
        value = DETERMINER_PREFIX_RE.sub("", value).strip()
        value = TOPIC_PREFIX_RE.sub("", value).strip()
    value = re.sub(r"(?:的)?(?:文章|内容|回答|说明)$", "", value).strip()
    return value.strip("：:，,。！？!?；;、 \t\r\n\"'")


def prompt_term_candidates(value: str) -> set[str]:
    """Extract readable prompt phrases without arbitrary CJK character windows."""
    raw_candidates: set[str] = set()
    raw_candidates.update(
        re.findall(r"[“\"《]([^”\"》]{2,24})[”\"》]", value)
    )
    raw_candidates.update(match.group(1) for match in TOPIC_MARKER_RE.finditer(value))
    raw_candidates.update(
        re.split(
            r"[，。！？；、,!?;:\n\r]+|(?:以及|或者|并且|同时|例如|比如|包括|然后|但是|而且)",
            value,
        )
    )
    raw_candidates.update(
        re.findall(
            r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_+#.\-]*"
            r"(?:\s+[A-Za-z][A-Za-z0-9_+#.\-]*)?(?![A-Za-z0-9_])",
            value,
        )
    )

    expanded_candidates = list(raw_candidates)
    for candidate in list(raw_candidates):
        candidate = clean_prompt_term(candidate)
        if "和" in candidate:
            parts = candidate.split("和")
            if len(parts) == 2 and all(2 <= len(part) <= 8 for part in parts):
                expanded_candidates.extend(parts)

    terms: set[str] = set()
    for candidate in expanded_candidates:
        candidate = clean_prompt_term(candidate)
        compact = re.sub(r"\s+", " ", candidate)
        compact_length = len(compact.replace(" ", ""))
        maximum_length = 12 if re.search(r"[\u3400-\u9fff]", compact) else 24
        if not 2 <= compact_length <= maximum_length:
            continue
        lowered = compact.lower()
        if lowered in GENERIC_TERMS or any(term in compact for term in GENERIC_TERMS):
            continue
        if re.match(r"^[A-Za-z]\.\s*", compact) or re.fullmatch(r"[A-Za-z]\.?,?", compact):
            continue
        if re.search(r"[\u3400-\u9fffA-Za-z]", compact):
            terms.add(compact)
    return terms


def build_term_document_frequency(rows: list[dict[str, Any]]) -> Counter[str]:
    frequency: Counter[str] = Counter()
    for row in rows:
        frequency.update(prompt_term_candidates(row["messages"][0]["content"]))
    return frequency


def source_terms(
    row: dict[str, Any], term_frequency: Counter[str], document_count: int
) -> list[str]:
    prompt = row["messages"][0]["content"]
    candidates = prompt_term_candidates(prompt)
    metadata = row["metadata"]
    metadata_terms = {
        re.sub(r"[\s,，。；;：:、]+", "", str(value)).strip()
        for value in list(metadata.get("ability") or [])
        + list(metadata.get("category") or [])
    }

    def score(term: str) -> tuple[float, int, str]:
        inverse_frequency = math.log(
            (document_count + 1) / (term_frequency[term] + 1)
        )
        metadata_boost = 2.0 if term in metadata_terms else 0.0
        return metadata_boost + inverse_frequency + 0.15 * len(term), len(term), term

    ranked = sorted(candidates, key=score, reverse=True)
    return ranked[:12]


def answer_end_phrase(answer: str, terms: list[str]) -> str:
    plain = re.sub(r"```.*?```", " ", answer, flags=re.DOTALL)
    candidates = [
        re.sub(r"^[#>*\-\d.\s]+|[。！？!?\s]+$", "", part).strip()
        for part in re.split(r"[。！？!?\n]", plain)
    ]
    candidates = [
        value
        for value in candidates
        if 8 <= len(value) <= 36
        and not value.endswith(("：", ":"))
        and not value.startswith("以下")
    ]
    if candidates:
        return candidates[-1]
    if terms:
        return f"关于{terms[0]}的说明到此结束"
    return "以上说明到此结束"


def allowed_constraints(bucket: str, sampling: dict[str, Any]) -> set[str]:
    all_ids = set(sampling["constraints"])
    rule = sampling["task_bucket_allowed_constraints"][bucket]
    if isinstance(rule, list):
        return set(rule)
    exclusions = {
        "all": set(),
        "all_except_placeholders_and_json": {
            "detectable_content:number_placeholders",
            "detectable_format:json_format",
        },
        "all_except_json": {"detectable_format:json_format"},
        "all_except_json_and_repeat_prompt": {
            "detectable_format:json_format",
            "combination:repeat_prompt",
        },
        "all_except_placeholders": {"detectable_content:number_placeholders"},
    }
    if rule not in exclusions:
        raise ValueError(f"Unknown allowed-constraint rule: {rule}")
    return all_ids - exclusions[rule]


def compatible(candidate: str, selected: list[str], sampling: dict[str, Any]) -> bool:
    selected_set = set(selected)
    for group in sampling["mutually_exclusive_groups"]:
        group_set = set(group)
        if candidate in group_set and selected_set & group_set:
            return False
    forbidden_pairs = {frozenset(pair) for pair in sampling["forbidden_pairs"]}
    return all(frozenset((candidate, item)) not in forbidden_pairs for item in selected)


def weighted_choice(
    candidates: list[str], sampling: dict[str, Any], rng: random.Random
) -> str:
    weights = [float(sampling["constraints"][item]["weight"]) for item in candidates]
    target = rng.random() * sum(weights)
    running = 0.0
    for item, weight in zip(candidates, weights):
        running += weight
        if running >= target:
            return item
    return candidates[-1]


def choose_constraint_ids(
    bucket: str,
    count: int,
    current_request: str,
    has_source_terms: bool,
    sampling: dict[str, Any],
    rng: random.Random,
) -> list[str]:
    allowed = allowed_constraints(bucket, sampling)
    if not has_source_terms:
        allowed.difference_update(
            {
                "keywords:existence",
                "keywords:frequency",
                "keywords:forbidden_words",
            }
        )
    repeat_limit = sampling["constraints"]["combination:repeat_prompt"][
        "maximum_source_prompt_characters"
    ]
    if len(current_request) > repeat_limit:
        allowed.discard("combination:repeat_prompt")

    maximum_attempts = int(sampling["maximum_sampling_attempts_per_prompt"])
    for _ in range(maximum_attempts):
        selected: list[str] = []
        for _ in range(count):
            candidates = sorted(
                item
                for item in allowed - set(selected)
                if compatible(item, selected, sampling)
            )
            if not candidates:
                break
            selected.append(weighted_choice(candidates, sampling, rng))
        if len(selected) == count:
            return selected
    raise ValueError(
        f"Cannot sample {count} compatible constraints for {bucket} "
        f"after {maximum_attempts} attempts"
    )


def constraint_instance(
    instruction_id: str,
    row: dict[str, Any],
    current_request: str,
    spec: dict[str, Any],
    term_frequency: Counter[str],
    document_count: int,
    rng: random.Random,
) -> tuple[dict[str, Any], str]:
    terms = source_terms(row, term_frequency, document_count)
    answer = row["messages"][1]["content"]

    if instruction_id == "keywords:existence":
        count = rng.randint(*spec["keyword_count"])
        keywords = rng.sample(terms, k=min(count, len(terms)))
        kwargs = {"keywords": keywords}
        text = spec["template"].format(keywords="、".join(keywords))
    elif instruction_id == "keywords:frequency":
        keyword = rng.choice(terms)
        frequency = 1 if len(keyword) > 8 else rng.randint(*spec["frequency"])
        kwargs = {"keyword": keyword, "frequency": frequency, "relation": spec["relation"]}
        text = spec["template"].format(keyword=keyword, frequency=frequency)
    elif instruction_id == "keywords:forbidden_words":
        count = rng.randint(*spec["forbidden_word_count"])
        words = rng.sample(terms, k=min(count, len(terms)))
        kwargs = {"forbidden_words": words}
        text = spec["template"].format(forbidden_words="、".join(words))
    elif instruction_id in {
        "length_constraints:number_sentences",
        "length_constraints:number_words",
    }:
        relation = rng.choice(["less_than", "at_least"])
        value = rng.randint(*spec[relation])
        argument = "num_sentences" if instruction_id.endswith("sentences") else "num_words"
        kwargs = {argument: value, "relation": relation.replace("_", " ")}
        text = spec["templates"][relation].format(**{argument: value})
    elif instruction_id == "length_constraints:number_paragraphs":
        value = rng.randint(*spec["num_paragraphs"])
        kwargs = {"num_paragraphs": value}
        text = spec["template"].format(num_paragraphs=value)
    elif instruction_id == "length_constraints:nth_paragraph_first_word":
        paragraphs = rng.randint(*spec["num_paragraphs"])
        nth = rng.randint(1, paragraphs)
        first_word = rng.choice(spec["first_words"])
        kwargs = {
            "num_paragraphs": paragraphs,
            "nth_paragraph": nth,
            "first_word": first_word,
        }
        text = spec["template"].format(
            num_paragraphs=paragraphs, nth_paragraph=nth, first_word=first_word
        )
    elif instruction_id == "detectable_content:number_placeholders":
        value = rng.randint(*spec["num_placeholders"])
        kwargs = {"num_placeholders": value}
        text = spec["template"].format(num_placeholders=value)
    elif instruction_id == "detectable_content:postscript":
        marker = rng.choice(spec["postscript_markers"])
        kwargs = {"postscript_marker": marker}
        text = spec["template"].format(postscript_marker=marker)
    elif instruction_id == "detectable_format:number_bullet_lists":
        value = rng.randint(*spec["num_bullets"])
        kwargs = {"num_bullets": value}
        text = spec["template"].format(num_bullets=value)
    elif instruction_id == "detectable_format:number_highlighted_sections":
        value = rng.randint(*spec["num_highlights"])
        kwargs = {"num_highlights": value}
        text = spec["template"].format(num_highlights=value)
    elif instruction_id == "detectable_format:multiple_sections":
        value = rng.randint(*spec["num_sections"])
        kwargs = {"section_spliter": spec["section_splitter"], "num_sections": value}
        text = spec["template"].format(num_sections=value)
    elif instruction_id == "combination:repeat_prompt":
        kwargs = {"prompt_to_repeat": current_request}
        text = spec["template"]
    elif instruction_id == "startend:end_checker":
        phrase = answer_end_phrase(answer, terms)
        kwargs = {"end_phrase": phrase}
        text = spec["template"].format(end_phrase=phrase)
    else:
        kwargs = {}
        text = spec["template"]
    return kwargs, text


def build_messages(
    row: dict[str, Any], context_turns: int, constraint_texts: list[str], config: dict[str, Any]
) -> tuple[list[dict[str, str]], str]:
    source_user = row["messages"][0]
    source_answer = row["messages"][1]
    if context_turns == 1:
        messages: list[dict[str, str]] = []
        current_request = source_user["content"]
    else:
        messages = [source_user, source_answer]
        construction = config["data"]["context_construction"]
        if context_turns == 3:
            messages.extend(
                [
                    {"role": "user", "content": construction["bridge_user"]},
                    {"role": "assistant", "content": construction["bridge_assistant"]},
                ]
            )
        current_request = "请在保留上面回答主题和核心信息的前提下重新作答。"

    numbered = "\n".join(
        f"{index}. {value}" for index, value in enumerate(constraint_texts, start=1)
    )
    current_content = f"{current_request}\n\n请同时满足以下要求：\n{numbered}"
    messages.append({"role": "user", "content": current_content})
    return messages, current_request


def load_official_checker() -> Any:
    path = PROJECT_DIR / "third_party" / "Multi-IF" / "ifeval.py"
    spec = importlib.util.spec_from_file_location("rlvr_official_ifeval", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import official checker from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_checker_args(rows: list[dict[str, Any]]) -> Counter[str]:
    ifeval = load_official_checker()
    counts: Counter[str] = Counter()
    for row in rows:
        for instruction_id, kwargs in zip(row["instruction_ids"], row["kwargs"]):
            checker = ifeval.INSTRUCTION_DICT[instruction_id](instruction_id)
            checker.build_description(**kwargs)
            counts[instruction_id] += 1
    return counts


def sample_manual_review(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for bucket in sorted({row["metadata"]["task_bucket"] for row in rows}):
        row = next(item for item in rows if item["metadata"]["task_bucket"] == bucket)
        selected.append(row)
        selected_ids.add(row["id"])
    for instruction_id in sorted({item for row in rows for item in row["instruction_ids"]}):
        row = next(
            item
            for item in rows
            if instruction_id in item["instruction_ids"] and item["id"] not in selected_ids
        )
        selected.append(row)
        selected_ids.add(row["id"])
    remaining = [row for row in rows if row["id"] not in selected_ids]
    random.Random(seed).shuffle(remaining)
    selected.extend(remaining[: size - len(selected)])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data = config["data"]
    output_paths = [
        PROJECT_DIR / data["train_output"],
        PROJECT_DIR / data["validation_output"],
        PROJECT_DIR / data["manual_review_output"],
        PROJECT_DIR / data["manifest_output"],
        PROJECT_DIR / data["audit_output"],
    ]
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "Refusing to overwrite RLVR outputs: "
            + ", ".join(str(path) for path in existing)
        )

    multi_if_paths = [
        PROJECT_DIR / config["evaluation_boundary"]["dev"],
        PROJECT_DIR / config["evaluation_boundary"]["test"],
    ]
    multi_if_prompts = load_multi_if_prompts(multi_if_paths)
    source_path = PROJECT_DIR / data["source"]
    selected, source_stats = select_sources(
        read_jsonl(source_path), config, multi_if_prompts
    )

    total = data["train_size"] + data["validation_size"]
    context_schedule = expanded_schedule(
        allocation(total, data["context_turn_ratios"]),
        random.Random(f"{config['seed']}:context"),
    )
    constraint_count_schedule = expanded_schedule(
        allocation(total, config["constraint_sampling"]["constraints_per_prompt_ratios"]),
        random.Random(f"{config['seed']}:constraint-count"),
    )

    sampling = config["constraint_sampling"]
    term_frequency = build_term_document_frequency(selected)
    rows: list[dict[str, Any]] = []
    for index, (source, context_turns, constraint_count) in enumerate(
        zip(selected, context_schedule, constraint_count_schedule)
    ):
        rng = random.Random(f"{config['seed']}:{source['metadata']['id']}:{index}")
        current_request = (
            source["messages"][0]["content"]
            if context_turns == 1
            else "请在保留上面回答主题和核心信息的前提下重新作答。"
        )
        instruction_ids = choose_constraint_ids(
            source["metadata"]["task_bucket"],
            int(constraint_count),
            current_request,
            bool(source_terms(source, term_frequency, len(selected))),
            sampling,
            rng,
        )
        kwargs_values: list[dict[str, Any]] = []
        constraint_texts: list[str] = []
        for instruction_id in instruction_ids:
            kwargs, text = constraint_instance(
                instruction_id,
                source,
                current_request,
                sampling["constraints"][instruction_id],
                term_frequency,
                len(selected),
                rng,
            )
            kwargs_values.append(kwargs)
            constraint_texts.append(text)
        messages, _ = build_messages(source, int(context_turns), constraint_texts, config)
        rows.append(
            {
                "id": "pending",
                "messages": messages,
                "instruction_ids": instruction_ids,
                "kwargs": kwargs_values,
                "constraint_categories": sorted(
                    {value.split(":", maxsplit=1)[0] for value in instruction_ids}
                ),
                "metadata": {
                    "source_id": str(source["metadata"]["id"]),
                    "task_bucket": source["metadata"]["task_bucket"],
                    "quality_score": source["metadata"]["quality_score"],
                    "context_turns": int(context_turns),
                    "constraint_count": int(constraint_count),
                },
            }
        )

    random.Random(f"{config['seed']}:final-split").shuffle(rows)
    for index, row in enumerate(rows, start=1):
        row["id"] = f"rlvr-{index:04d}"

    checker_counts = validate_checker_args(rows)
    final_prompt_overlap = sum(
        normalize_prompt(row["messages"][-1]["content"]) in multi_if_prompts
        for row in rows
    )
    if final_prompt_overlap:
        raise AssertionError(f"Constructed prompts overlap Multi-IF: {final_prompt_overlap}")

    train_rows = rows[: data["train_size"]]
    validation_rows = rows[data["train_size"] :]
    review_rows = sample_manual_review(rows, data["manual_audit_size"], config["seed"])
    train_path, validation_path, review_path, manifest_path, audit_path = output_paths
    write_jsonl(train_path, train_rows)
    write_jsonl(validation_path, validation_rows)
    write_jsonl(review_path, review_rows)

    manifest = {
        "schema_version": "rlvr_constraint_data/v1",
        "status": "pending_manual_review",
        "seed": config["seed"],
        "config": {"path": args.config.as_posix(), "sha256": sha256_file(args.config)},
        "source": {"path": data["source"], "sha256": sha256_file(source_path)},
        "outputs": {
            "train": {"path": data["train_output"], "rows": len(train_rows), "sha256": sha256_file(train_path)},
            "validation": {"path": data["validation_output"], "rows": len(validation_rows), "sha256": sha256_file(validation_path)},
            "manual_review": {"path": data["manual_review_output"], "rows": len(review_rows), "sha256": sha256_file(review_path)},
        },
        "counts": {
            "task_buckets": dict(sorted(Counter(row["metadata"]["task_bucket"] for row in rows).items())),
            "context_turns": dict(sorted(Counter(str(row["metadata"]["context_turns"]) for row in rows).items())),
            "constraints_per_prompt": dict(sorted(Counter(str(row["metadata"]["constraint_count"]) for row in rows).items())),
            "instruction_ids": dict(sorted(checker_counts.items())),
        },
        "checks": {
            **source_stats,
            "source_ids_unique": len({row["metadata"]["source_id"] for row in rows}),
            "output_ids_unique": len({row["id"] for row in rows}),
            "final_prompt_overlap_with_multi_if": final_prompt_overlap,
            "checker_argument_errors": 0,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        "# RLVR Constraint Data Audit\n\n"
        "Status: pending 100-row manual review. Do not train T1 or RLVR yet.\n\n"
        f"- Train rows: {len(train_rows)}\n"
        f"- Validation rows: {len(validation_rows)}\n"
        f"- Unique source IDs: {manifest['checks']['source_ids_unique']}\n"
        f"- Multi-IF final-prompt overlap: {final_prompt_overlap}\n"
        f"- Official checker argument errors: 0\n"
        f"- Manual review queue: `{data['manual_review_output']}`\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
