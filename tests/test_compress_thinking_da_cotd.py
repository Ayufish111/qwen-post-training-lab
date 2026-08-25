"""CPU-only tests for adaptive T1 reasoning compression planning."""

from __future__ import annotations

import copy
import unittest

from scripts.compress_thinking_da_cotd import (
    BudgetDecision,
    build_review_queue,
    clean_editor_output,
    compressed_row,
    interpolate_budget,
    original_reasoning_answer,
    plan_summary,
    reasoning_copies_fixed_answer,
    structural_percentile,
    structural_score,
)


def make_row(
    row_id: str = "rlvr-test",
    *,
    constraints: int = 2,
    turns: int = 1,
    categories: int = 2,
    reasoning: str = "先识别约束，再检查格式。",
    answer: str = "最终答案",
    attempts: int = 1,
) -> dict:
    category_values = [f"category-{index}" for index in range(categories)]
    instruction_ids = [f"constraint-{index}" for index in range(constraints)]
    return {
        "id": row_id,
        "messages": [
            {"role": "user", "content": "测试问题"},
            {
                "role": "assistant",
                "reasoning_content": reasoning,
                "content": answer,
            },
        ],
        "instruction_ids": instruction_ids,
        "kwargs": [{} for _ in instruction_ids],
        "constraint_categories": category_values,
        "metadata": {
            "constraint_count": constraints,
            "context_turns": turns,
        },
        "teacher": {
            "reasoning_content": reasoning,
            "content": answer,
            "reasoning_tokens": 20,
            "generated_tokens": 40,
            "attempts": attempts,
        },
    }


class StructuralDifficultyTest(unittest.TestCase):
    def test_score_excludes_teacher_retries(self) -> None:
        first_pass = make_row(attempts=1)
        retried = make_row(attempts=3)
        self.assertEqual(structural_score(first_pass), structural_score(retried))

    def test_percentile_is_monotonic(self) -> None:
        reference = [1, 1, 2, 3, 3, 5]
        percentiles = [structural_percentile(score, reference) for score in range(1, 6)]
        self.assertEqual(percentiles, sorted(percentiles))
        self.assertGreater(percentiles[-1], percentiles[0])


class BudgetInterpolationTest(unittest.TestCase):
    def test_anchor_values_are_preserved(self) -> None:
        kwargs = {"easy": 128, "medium": 256, "hard": 512, "round_to": 32}
        self.assertEqual(interpolate_budget(0.0, **kwargs), 128)
        self.assertEqual(interpolate_budget(0.5, **kwargs), 256)
        self.assertEqual(interpolate_budget(1.0, **kwargs), 512)

    def test_interpolation_rounds_up_to_configured_multiple(self) -> None:
        budget = interpolate_budget(
            0.25, easy=128, medium=256, hard=512, round_to=32
        )
        self.assertEqual(budget, 192)
        rounded = interpolate_budget(
            0.30, easy=128, medium=256, hard=512, round_to=32
        )
        self.assertEqual(rounded, 224)

    def test_plan_lists_only_over_budget_example_ids(self) -> None:
        keep = make_row(row_id="keep")
        keep["teacher"]["reasoning_tokens"] = 100
        rewrite = make_row(row_id="rewrite")
        rewrite["teacher"]["reasoning_tokens"] = 600
        plan = plan_summary(
            [keep, rewrite],
            [structural_score(keep), structural_score(rewrite)],
            {"easy": 128, "medium": 256, "hard": 512},
            32,
        )
        self.assertEqual(plan["rewrite_example_ids"], ["rewrite"])


class DataIntegrityTest(unittest.TestCase):
    def test_original_answer_is_not_stripped(self) -> None:
        row = make_row(reasoning="  推理正文\n", answer="  最终答案\n")
        reasoning, answer = original_reasoning_answer(row)
        self.assertEqual(reasoning, "  推理正文\n")
        self.assertEqual(answer, "  最终答案\n")

    def test_compressed_row_keeps_answer_byte_identical(self) -> None:
        row = make_row(answer="  最终答案\n")
        before = copy.deepcopy(row)
        result = compressed_row(
            row,
            "精简推理",
            row["teacher"]["content"],
            BudgetDecision(3, 0.5, 256, "medium"),
            original_tokens=20,
            compressed_tokens=4,
            method="teacher_rewrite",
            editor_attempts=1,
            editor_model="Qwen/Qwen3-4B",
            strict_results=[True, True],
        )
        self.assertEqual(result["messages"][-1]["content"], before["teacher"]["content"])
        self.assertEqual(row, before)
        self.assertTrue(result["compression"]["answer_unchanged"])


class EditorOutputTest(unittest.TestCase):
    def test_cleans_explicit_thinking_wrapper(self) -> None:
        self.assertEqual(
            clean_editor_output("<think>保留关键步骤</think><|im_end|>"),
            "保留关键步骤",
        )

    def test_cleans_implicit_opening_wrapper(self) -> None:
        self.assertEqual(
            clean_editor_output("保留关键步骤</think><|im_end|>"),
            "保留关键步骤",
        )

    def test_cleans_label_and_code_fence(self) -> None:
        self.assertEqual(
            clean_editor_output("```text\n压缩后的推理：关键步骤\n```"),
            "关键步骤",
        )

    def test_removes_only_standalone_leading_answer_title(self) -> None:
        self.assertEqual(
            clean_editor_output("<<作者判断>>\n任务判断 -> 核对作者"),
            "任务判断 -> 核对作者",
        )
        self.assertEqual(
            clean_editor_output("格式检查：最终回答需要<<标题>>。"),
            "格式检查：最终回答需要<<标题>>。",
        )

    def test_detects_substantial_fixed_answer_copy(self) -> None:
        answer = "这是一个足够长的固定最终答案，其中还包含必须满足的格式与核心结论。"
        reasoning = "前置说明\n" + answer + "\n后置说明"
        self.assertTrue(reasoning_copies_fixed_answer(reasoning, answer))

    def test_short_conclusion_can_appear_in_reasoning(self) -> None:
        self.assertFalse(reasoning_copies_fixed_answer("因此选择 B。", "B"))


class ReviewQueueTest(unittest.TestCase):
    def test_queue_is_deterministic_and_stratified(self) -> None:
        rows = []
        for band_index, band in enumerate(("easy", "hard")):
            for method_index, method in enumerate(("keep", "teacher_rewrite")):
                for sample_index in range(3):
                    row = make_row(
                        row_id=f"{band_index}-{method_index}-{sample_index}"
                    )
                    row["compression"] = {
                        "budget_band": band,
                        "method": method,
                        "budget_tokens": 128 if band == "easy" else 512,
                        "original_reasoning_tokens": 20,
                        "compressed_reasoning_tokens": 10,
                    }
                    rows.append(row)

        first = build_review_queue(rows, per_stratum=2, seed=42)
        second = build_review_queue(list(reversed(rows)), per_stratum=2, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)
        strata = {
            (item["stratum"]["budget_band"], item["stratum"]["method"])
            for item in first
        }
        self.assertEqual(len(strata), 4)


if __name__ == "__main__":
    unittest.main()
