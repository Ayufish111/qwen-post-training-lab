"""Interactively audit 100 deterministic samples from the full SFT dataset."""

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


SEED = 42
SAMPLE_SIZE = 100

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "sft" / "full_clean_10000.jsonl"
REPORT_PATH = PROJECT_DIR / "reports" / "sft_data_audit.md"
PROGRESS_PATH = PROJECT_DIR / "reports" / ".sft_audit_progress.json"
START_MARKER = "<!-- MANUAL_AUDIT_START -->"
END_MARKER = "<!-- MANUAL_AUDIT_END -->"


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def sample_key(row):
    question = row["messages"][0]["content"]
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def select_samples(rows):
    """Cover every task bucket, then fill the sample deterministically."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["metadata"]["task_bucket"]].append(row)

    selected = []
    selected_keys = set()
    for bucket in sorted(groups):
        group = sorted(groups[bucket], key=sample_key)
        row = random.Random(f"{SEED}:{bucket}").choice(group)
        selected.append(row)
        selected_keys.add(sample_key(row))

    remaining = [row for row in rows if sample_key(row) not in selected_keys]
    random.Random(SEED).shuffle(remaining)
    selected.extend(remaining[: SAMPLE_SIZE - len(selected)])
    random.Random(SEED + 1).shuffle(selected)
    return selected


def load_progress():
    if not PROGRESS_PATH.exists():
        return {}
    return json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))


def save_progress(progress):
    PROGRESS_PATH.write_text(
        json.dumps(progress, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_report(samples, progress):
    counts = Counter(item["decision"] for item in progress.values())
    task_counts = Counter(row["metadata"]["task_bucket"] for row in samples)
    task_text = ", ".join(
        f"{task}={count}" for task, count in sorted(task_counts.items())
    )

    reviewed_issues = []
    for index, row in enumerate(samples, start=1):
        result = progress[sample_key(row)]
        if result["decision"] == "pass":
            continue
        question = " ".join(row["messages"][0]["content"].split())[:120]
        reason = result.get("reason") or "No reason recorded"
        reviewed_issues.append(
            f"- #{index} `{result['decision']}` ({row['metadata']['task_bucket']}): "
            f"{reason}; prompt: {question}"
        )
    issue_text = "\n".join(reviewed_issues) or "- No minor or major issues recorded."

    section = f"""{START_MARKER}
## Manual stratified audit

- Method: seed {SEED}; one guaranteed sample per task bucket, then seeded random fill.
- Reviewed: {len(samples)} rows from `full_clean_10000.jsonl`.
- Task coverage: {task_text}.
- Result: pass={counts['pass']}, minor={counts['minor']}, major={counts['major']}.

Observed non-pass cases:

{issue_text}

This is a sampled quality audit, not a claim that all 10,000 rows were manually
cleaned or fact-checked.
{END_MARKER}"""

    report = REPORT_PATH.read_text(encoding="utf-8")
    if START_MARKER in report and END_MARKER in report:
        before = report.split(START_MARKER, 1)[0].rstrip()
        after = report.split(END_MARKER, 1)[1].lstrip()
        report = before + "\n\n" + section + "\n\n" + after
    else:
        report = report.rstrip() + "\n\n" + section + "\n"
    REPORT_PATH.write_text(report, encoding="utf-8")


def main():
    samples = select_samples(read_jsonl(DATA_PATH))
    progress = load_progress()

    print("审核标准：1=pass，2=minor，3=major，q=保存并退出")
    print("pass: 正确且遵循指令")
    print("minor: 基本可用，但有小问题")
    print("major: 错误、编造、答非所问或未遵循指令")

    for index, row in enumerate(samples, start=1):
        key = sample_key(row)
        if key in progress:
            continue

        metadata = row["metadata"]
        print("\n" + "=" * 80)
        print(f"样本 {index}/{SAMPLE_SIZE} | 已完成 {len(progress)}/{SAMPLE_SIZE}")
        print("任务类型：", metadata["task_bucket"])
        print("质量分数：", metadata["quality_score"])
        print("风险标签：", metadata["quality_flags"])
        print("\n问题：\n", row["messages"][0]["content"])
        print("\n回答：\n", row["messages"][1]["content"])

        while True:
            choice = input("\n判断 [1/2/3/q]：").strip().lower()
            if choice == "q":
                save_progress(progress)
                print("进度已保存：", PROGRESS_PATH)
                return
            if choice in {"1", "2", "3"}:
                break
            print("请输入 1、2、3 或 q。")

        decision = {"1": "pass", "2": "minor", "3": "major"}[choice]
        reason = ""
        if decision != "pass":
            reason = input("简要原因：").strip()
        progress[key] = {"decision": decision, "reason": reason}
        save_progress(progress)

    write_report(samples, progress)
    PROGRESS_PATH.unlink(missing_ok=True)
    print("\n100 条审核完成，结果已写入：", REPORT_PATH)


if __name__ == "__main__":
    main()
