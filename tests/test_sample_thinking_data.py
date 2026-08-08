"""CPU-only tests for the T1 teacher-generation contract."""

import unittest

from scripts.sample_thinking_data import difficulty_bucket, parse_thinking_continuation


class SampleThinkingDataTest(unittest.TestCase):
    def test_parser_accepts_qwen3_prompt_opening_marker(self) -> None:
        parsed = parse_thinking_continuation("先列出约束。\n</think>\n最终答案")
        self.assertEqual(parsed, ("先列出约束。", "最终答案"))

    def test_parser_accepts_explicit_opening_and_removes_end_marker(self) -> None:
        parsed = parse_thinking_continuation("<think>step one</think>answer<|im_end|>")
        self.assertEqual(parsed, ("step one", "answer"))

    def test_parser_rejects_missing_or_empty_sections(self) -> None:
        self.assertIsNone(parse_thinking_continuation("answer only"))
        self.assertIsNone(parse_thinking_continuation("reasoning</think>"))
        self.assertIsNone(parse_thinking_continuation("</think>answer"))

    def test_difficulty_uses_only_training_metadata(self) -> None:
        easy = {"instruction_ids": ["detectable_format:title"], "constraint_categories": ["detectable_format"], "metadata": {"constraint_count": 1, "context_turns": 1}}
        hard = {"instruction_ids": ["a", "b", "c"], "constraint_categories": ["a", "b", "c"], "metadata": {"constraint_count": 3, "context_turns": 3}}
        self.assertEqual(difficulty_bucket(easy), "easy")
        self.assertEqual(difficulty_bucket(hard), "hard")
        self.assertEqual(difficulty_bucket(easy, first_pass=False, attempts=2), "medium")


if __name__ == "__main__":
    unittest.main()
