from __future__ import annotations

import unittest

from src.build_distill_dataset import (
    DistillDataError,
    has_severe_repetition_loop,
    split_tokenized_rows,
    tokenize_row,
)


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
        return_dict,
    ):
        self.assertions = (tokenize, enable_thinking, return_dict)
        if add_generation_prompt:
            ids = [101, 102, 103]
        else:
            ids = [101, 102, 103, 201, 202, 203]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def make_row(row_id: str = "rlvr-0001") -> dict:
    return {
        "id": row_id,
        "messages": [
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "reasoning_content": "reasoning",
                "content": "answer",
            },
        ],
        "teacher": {
            "raw_continuation": "<think>reasoning</think>answer",
            "reasoning_content": "reasoning",
            "content": "answer",
            "strict_results": [True],
            "truncated": False,
        },
    }


class TokenizeDistillRowTest(unittest.TestCase):
    def test_masks_prompt_and_keeps_assistant_targets(self) -> None:
        tokenizer = FakeTokenizer()
        result = tokenize_row(make_row(), tokenizer, max_length=16)
        self.assertEqual(result["input_ids"], [101, 102, 103, 201, 202, 203])
        self.assertEqual(result["labels"], [-100, -100, -100, 201, 202, 203])
        self.assertEqual(result["attention_mask"], [1] * 6)
        self.assertEqual(tokenizer.assertions, (True, True, True))

    def test_rejects_empty_reasoning_or_answer(self) -> None:
        for field in ("reasoning_content", "content"):
            row = make_row()
            row["messages"][-1][field] = ""
            with self.assertRaises(DistillDataError):
                tokenize_row(row, FakeTokenizer())

    def test_rejects_overlength_without_truncation(self) -> None:
        with self.assertRaisesRegex(DistillDataError, "exceeds max_length"):
            tokenize_row(make_row(), FakeTokenizer(), max_length=5)

    def test_rejects_failed_or_incomplete_teacher_audit(self) -> None:
        failed = make_row()
        failed["teacher"]["strict_results"] = [False]
        with self.assertRaisesRegex(DistillDataError, "constraint check failed"):
            tokenize_row(failed, FakeTokenizer())

        incomplete = make_row()
        incomplete["teacher"]["raw_continuation"] = "reasoning without delimiter"
        with self.assertRaisesRegex(DistillDataError, "reasoning delimiter"):
            tokenize_row(incomplete, FakeTokenizer())

    def test_detects_only_severe_consecutive_repetition(self) -> None:
        repeated = "这是一个足够长的重复推理单元。\n" * 3
        self.assertTrue(has_severe_repetition_loop(repeated))
        self.assertFalse(
            has_severe_repetition_loop("先检查格式。\n再生成回答。\n最后复查约束。")
        )


class DistillSplitTest(unittest.TestCase):
    def test_split_is_reproducible_and_disjoint(self) -> None:
        rows = [{"id": f"row-{index}"} for index in range(100)]
        train_a, validation_a = split_tokenized_rows(rows, seed=42)
        train_b, validation_b = split_tokenized_rows(rows, seed=42)
        self.assertEqual(train_a, train_b)
        self.assertEqual(validation_a, validation_b)
        self.assertEqual(len(train_a), 95)
        self.assertEqual(len(validation_a), 5)
        self.assertFalse(
            {row["id"] for row in train_a}
            & {row["id"] for row in validation_a}
        )


if __name__ == "__main__":
    unittest.main()
