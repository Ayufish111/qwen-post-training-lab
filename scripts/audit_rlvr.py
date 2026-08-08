"""Interactively review the frozen 100-row RLVR constraint audit queue."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
QUEUE_PATH = PROJECT_DIR / "data" / "rlvr" / "manual_review_queue_100.jsonl"
MANIFEST_PATH = PROJECT_DIR / "reports" / "manifests" / "rlvr_constraint_data_manifest.json"
AUDIT_PATH = PROJECT_DIR / "reports" / "rlvr" / "constraint_data_audit.md"
REPORT_PATH = PROJECT_DIR / "reports" / "rlvr" / "constraint_manual_review.md"
PROGRESS_PATH = PROJECT_DIR / "tmp" / "rlvr_manual_review_progress.json"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_progress() -> dict[str, dict[str, str]]:
    if not PROGRESS_PATH.exists():
        return {}
    return json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))


def save_progress(progress: dict[str, dict[str, str]]) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(
        json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def print_row(index: int, total: int, row: dict, completed: int) -> None:
    metadata = row["metadata"]
    print("\n" + "=" * 88)
    print(
        f"样本 {index}/{total} | 已完成 {completed}/{total} | {row['id']} | "
        f"{metadata['task_bucket']} | 上下文 {metadata['context_turns']} 轮"
    )
    print("来源 ID：", metadata["source_id"], "质量分数：", metadata["quality_score"])
    print("\n消息：")
    for message in row["messages"]:
        print(f"\n[{message['role']}]\n{message['content']}")
    print("\n约束参数：")
    for instruction_id, kwargs in zip(row["instruction_ids"], row["kwargs"]):
        print(f"- {instruction_id}: {json.dumps(kwargs, ensure_ascii=False)}")
    print(
        "\n判断标准：1=pass（自然、可满足、可验证）；"
        "2=minor（可训练但有轻微低价值/表达问题）；"
        "3=major（冲突、不可满足、乱码/断词、明显答非所问）；q=保存退出"
    )


def write_report(rows: list[dict], progress: dict[str, dict[str, str]]) -> None:
    counts = Counter(item["decision"] for item in progress.values())
    issues = []
    for index, row in enumerate(rows, start=1):
        result = progress.get(row["id"])
        if not result or result["decision"] == "pass":
            continue
        reason = result.get("reason") or "未记录原因"
        issues.append(
            f"- `{row['id']}` ({row['metadata']['task_bucket']}): "
            f"`{result['decision']}`；{reason}"
        )
    issue_text = "\n".join(issues) or "- 无 minor/major 条目。"
    REPORT_PATH.write_text(
        "# RLVR 100-row Manual Review\n\n"
        f"- Reviewed: {len(progress)}/{len(rows)}\n"
        f"- pass: {counts['pass']}\n"
        f"- minor: {counts['minor']}\n"
        f"- major: {counts['major']}\n\n"
        "## Non-pass Items\n\n"
        f"{issue_text}\n",
        encoding="utf-8",
    )


def approve(rows: list[dict], progress: dict[str, dict[str, str]]) -> None:
    if len(progress) != len(rows):
        raise RuntimeError("Cannot approve before all 100 rows are reviewed.")
    counts = Counter(item["decision"] for item in progress.values())
    if counts["major"]:
        raise RuntimeError("Cannot approve while major review items remain.")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["status"] = "approved_for_t1_and_rlvr"
    manifest["manual_review"] = {
        "report": REPORT_PATH.as_posix(),
        "rows": len(rows),
        "counts": dict(counts),
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    audit = AUDIT_PATH.read_text(encoding="utf-8")
    audit = audit.replace(
        "Status: pending 100-row manual review. Do not train T1 or RLVR yet.",
        "Status: approved_for_t1_and_rlvr after completed 100-row manual review.",
    )
    AUDIT_PATH.write_text(audit, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Approve only after every row is reviewed and no major item remains.",
    )
    args = parser.parse_args()

    rows = read_jsonl(QUEUE_PATH)
    progress = load_progress()
    if args.approve:
        write_report(rows, progress)
        approve(rows, progress)
        PROGRESS_PATH.unlink(missing_ok=True)
        print("RLVR manual review approved; T1/RLVR may proceed.")
        return

    print("RLVR 人工审查：1=pass，2=minor，3=major，q=保存并退出")
    for index, row in enumerate(rows, start=1):
        if row["id"] in progress:
            continue
        print_row(index, len(rows), row, len(progress))
        while True:
            choice = input("\n判断 [1/2/3/q]：").strip().lower()
            if choice == "q":
                save_progress(progress)
                write_report(rows, progress)
                print("进度已保存：", PROGRESS_PATH)
                return
            if choice in {"1", "2", "3"}:
                break
            print("请输入 1、2、3 或 q。")
        decision = {"1": "pass", "2": "minor", "3": "major"}[choice]
        reason = ""
        if decision != "pass":
            reason = input("简要原因：").strip()
        progress[row["id"]] = {"decision": decision, "reason": reason}
        save_progress(progress)
        write_report(rows, progress)


if __name__ == "__main__":
    main()
