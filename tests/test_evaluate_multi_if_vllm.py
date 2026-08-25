import sys
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate_multi_if_vllm import (
    extract_final_content,
    generate_with_answer_only_fallback,
    thinking_structure_ok,
)
import evaluate_multi_if_vllm_final as final


class VllmExtractionTest(unittest.TestCase):
    def test_extracts_answer_after_reasoning(self):
        self.assertEqual(extract_final_content("<think>hidden</think>answer<|im_end|>"), "answer")

    def test_unclosed_reasoning_is_not_returned_as_answer(self):
        self.assertEqual(extract_final_content("<think>hidden"), "")
        self.assertEqual(
            extract_final_content("hidden without opening marker", enable_thinking=True),
            "",
        )

    def test_completion_close_marker_uses_prompt_supplied_open_marker(self):
        self.assertEqual(
            extract_final_content("</think>final answer", enable_thinking=True),
            "final answer",
        )
        self.assertTrue(
            thinking_structure_ok("</think>final answer", enable_thinking=True)
        )

    def test_malformed_thinking_structure_is_not_scored(self):
        raw = "</think>final answer<think>unclosed reasoning"
        self.assertFalse(thinking_structure_ok(raw, enable_thinking=True))
        self.assertEqual(extract_final_content(raw, enable_thinking=True), "")

    def test_plain_answer_is_preserved(self):
        self.assertEqual(extract_final_content("plain answer"), "plain answer")

    def test_thinking_structure_is_audited(self):
        self.assertTrue(thinking_structure_ok("<think>hidden</think>answer"))
        self.assertTrue(thinking_structure_ok("plain answer"))
        self.assertFalse(thinking_structure_ok("<think>hidden"))
        self.assertFalse(thinking_structure_ok("answer</think>"))
        self.assertFalse(
            thinking_structure_ok("hidden without close", enable_thinking=True)
        )
class MergedModelLoadTest(unittest.TestCase):
    def test_merged_model_disables_dynamic_lora(self):
        class Tokenizer:
            pad_token_id = 151643

            def convert_tokens_to_ids(self, token):
                return {"<|im_end|>": 151645, "<|endoftext|>": 151643}.get(token, -1)

            def decode(self, token_ids, **kwargs):
                return "<think>hidden reasoning</think>final answer"

        class FakeVllm:
            class LLM:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs

        tokenizer = Tokenizer()
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
            with patch.object(final.importlib, "import_module", return_value=FakeVllm):
                _, loaded_tokenizer, llm, request = final.load_vllm(
                    "merged-model", None, tensor_parallel_size=1, gpu_memory_utilization=0.85
                )

        self.assertIs(loaded_tokenizer, tokenizer)
        self.assertIsNone(request)
        self.assertFalse(llm.kwargs["enable_lora"])
        self.assertNotIn("max_lora_rank", llm.kwargs)
class VllmGenerationParsingTest(unittest.TestCase):
    def test_generation_preserves_boundaries_and_returns_final_answer(self):
        class Tokenizer:
            pad_token_id = 151643

            def apply_chat_template(self, messages, **kwargs):
                return "prompt"

            def convert_tokens_to_ids(self, token):
                return {"<|im_end|>": 151645, "<|endoftext|>": 151643}.get(token, -1)

            def decode(self, token_ids, **kwargs):
                return "<think>hidden reasoning</think>final answer"

        class SamplingParams:
            last_kwargs = None

            def __init__(self, **kwargs):
                SamplingParams.last_kwargs = kwargs

        class FakeVllm:
            pass

        FakeVllm.SamplingParams = SamplingParams

        class Completion:
            text = "cleaned by vLLM"
            token_ids = (1, 2, 3)
            finish_reason = "stop"

        class RequestOutput:
            pass

        RequestOutput.outputs = [Completion()]

        class Llm:
            def generate(self, prompts, **kwargs):
                return [RequestOutput()]

        response, generation = final.generate_one(
            FakeVllm, Tokenizer(), Llm(), None, [{"role": "user", "content": "x"}], 1024, True
        )
        self.assertEqual(response, "final answer")
        self.assertFalse(SamplingParams.last_kwargs["skip_special_tokens"])
        self.assertFalse(generation["clipped"])
        self.assertTrue(generation["saw_thinking_tokens"])
class AnswerOnlyFallbackTest(unittest.TestCase):
    def test_retries_only_when_primary_answer_is_empty(self):
        calls = []

        def fake_generate(*args):
            enable_thinking = args[-1]
            calls.append(enable_thinking)
            if enable_thinking:
                return "", {
                    "generated_tokens": 1024,
                    "generation_seconds": 1.0,
                    "natural_eos": False,
                    "clipped": True,
                    "enable_thinking": True,
                }
            return "recovered answer", {
                "generated_tokens": 24,
                "generation_seconds": 0.2,
                "natural_eos": True,
                "clipped": False,
                "enable_thinking": False,
            }

        response, generation = generate_with_answer_only_fallback(
            fake_generate, None, None, None, None, [], 1024
        )
        self.assertEqual(response, "recovered answer")
        self.assertEqual(calls, [True, False])
        self.assertTrue(generation["answer_only_fallback_attempted"])
        self.assertTrue(generation["answer_only_fallback_used"])
        self.assertTrue(generation["primary_clipped"])
        self.assertEqual(generation["primary_generated_tokens"], 1024)

    def test_does_not_retry_nonempty_primary_answer(self):
        calls = []

        def fake_generate(*args):
            calls.append(args[-1])
            return "primary answer", {"clipped": False, "generated_tokens": 8}

        response, generation = generate_with_answer_only_fallback(
            fake_generate, None, None, None, None, [], 1024
        )
        self.assertEqual(response, "primary answer")
        self.assertEqual(calls, [True])
        self.assertFalse(generation["answer_only_fallback_attempted"])

    def test_retries_nonempty_clipped_primary_answer(self):
        calls = []

        def fake_generate(*args):
            enable_thinking = args[-1]
            calls.append(enable_thinking)
            if enable_thinking:
                return "internal reasoning", {
                    "clipped": True,
                    "saw_thinking_tokens": True,
                    "thinking_structure_ok": False,
                    "generated_tokens": 1024,
                }
            return "answer-only result", {
                "clipped": False,
                "saw_thinking_tokens": False,
                "thinking_structure_ok": True,
                "generated_tokens": 12,
            }

        response, generation = generate_with_answer_only_fallback(
            fake_generate, None, None, None, None, [], 1024
        )
        self.assertEqual(response, "answer-only result")
        self.assertEqual(calls, [True, False])
        self.assertTrue(generation["primary_clipped"])
        self.assertTrue(generation["primary_unclosed_thinking"])
