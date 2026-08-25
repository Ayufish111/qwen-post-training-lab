from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from src.rlvr_contract import (
    RewardBatchAdapter,
    completion_to_text,
    load_algorithm_contract,
    load_training_pool,
    to_trl_rows,
    validate_algorithm_matrix,
    validate_rlvr_config,
    validate_training_row,
)
from src.rlvr_rewards import RewardCalculator
from src.train_rlvr import (
    advantages_have_learning_signal,
    WeightedRepeatSampler,
    build_grpo_kwargs,
    build_preflight,
    configure_qwen_chat_termination,
    decode_reward_completions,
    evaluate_smoke_health,
    validate_output_state,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((PROJECT_DIR / "configs" / "rlvr.yaml").read_text(encoding="utf-8"))


class FakeInstruction:
    def __init__(self, instruction_id: str) -> None:
        self.instruction_id = instruction_id

    def build_description(self, **_: object) -> None:
        return None

    def check_following(self, answer: str) -> bool:
        return "pass" in answer


class FakeIfeval:
    INSTRUCTION_DICT = {
        "keywords:existence": FakeInstruction,
        "detectable_format:title": FakeInstruction,
    }


def make_row(row_id: str = "rlvr-test") -> dict:
    return {
        "id": row_id,
        "messages": [{"role": "user", "content": "question"}],
        "instruction_ids": ["keywords:existence"],
        "kwargs": [{}],
        "constraint_categories": ["keywords"],
        "metadata": {
            "source_id": "source",
            "task_bucket": "general_instruction",
            "quality_score": 60.0,
            "context_turns": 1,
            "constraint_count": 1,
        },
    }


class AlgorithmMatrixTest(unittest.TestCase):
    def test_frozen_config_is_valid(self) -> None:
        validate_rlvr_config(CONFIG)
        self.assertEqual(load_algorithm_contract(CONFIG, "R0").name, "grpo")
        self.assertNotIn("dynamic_sampling", CONFIG["rlvr"]["algorithms"]["R0"])
        self.assertFalse(load_algorithm_contract(CONFIG, "R1").constraint_aware_sampling)
        self.assertTrue(load_algorithm_contract(CONFIG, "R2").constraint_aware_sampling)

    def test_r1_r2_cannot_differ_in_reward_or_loss(self) -> None:
        config = copy.deepcopy(CONFIG)
        config["rlvr"]["algorithms"]["R2"]["soft_overlong_penalty"] = False
        with self.assertRaisesRegex(ValueError, "both enable soft_overlong_penalty"):
            validate_algorithm_matrix(config)

    def test_effective_batch_must_be_divisible_by_group_size(self) -> None:
        config = copy.deepcopy(CONFIG)
        config["rlvr"]["gradient_accumulation_steps"] = 7
        with self.assertRaisesRegex(ValueError, "divisible"):
            validate_rlvr_config(config)


class TrainingDataContractTest(unittest.TestCase):
    def test_row_and_trl_conversion(self) -> None:
        row = make_row()
        validate_training_row(row)
        converted = to_trl_rows([row])[0]
        self.assertEqual(converted["row_id"], row["id"])
        self.assertEqual(converted["prompt"], row["messages"])
        self.assertEqual(converted["constraint_kwargs"], row["kwargs"])
        self.assertNotIn("kwargs", converted)

    def test_rejects_wrong_final_role_and_category(self) -> None:
        row = make_row()
        row["messages"].append({"role": "assistant", "content": "answer"})
        with self.assertRaisesRegex(ValueError, "final message"):
            validate_training_row(row)

        row = make_row()
        row["constraint_categories"] = ["detectable_format"]
        with self.assertRaisesRegex(ValueError, "constraint_categories"):
            validate_training_row(row)

    def test_real_training_pool_is_complete_and_unique(self) -> None:
        rows = load_training_pool(
            PROJECT_DIR / CONFIG["rlvr"]["train_input"], expected_rows=2000
        )
        self.assertEqual(len(rows), 2000)
        self.assertEqual(len({row["id"] for row in rows}), 2000)


class RewardAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.calculator = RewardCalculator(FakeIfeval())

    def adapter(self, experiment_id: str, token_count: int = 4) -> RewardBatchAdapter:
        return RewardBatchAdapter(
            self.calculator,
            load_algorithm_contract(CONFIG, experiment_id),
            token_counter=lambda _: token_count,
            maximum_completion_tokens=10,
            overlong_buffer_tokens=2,
        )

    def test_normalizes_conversational_completion(self) -> None:
        self.assertEqual(completion_to_text("plain"), "plain")
        self.assertEqual(completion_to_text({"content": "dict"}), "dict")
        self.assertEqual(
            completion_to_text([{"role": "assistant", "content": "chat"}]),
            "chat",
        )

    def test_structured_qwen_reasoning_content_is_preserved(self) -> None:
        adapter = self.adapter("R0")
        values = adapter(
            [[{"role": "assistant", "reasoning_content": "计划", "content": "pass"}]],
            [["keywords:existence"]],
            [[{}]],
            ["structured"],
        )
        self.assertEqual(values, [1.0])
        self.assertTrue(adapter.records[0].parse_ok)

    def test_raw_qwen_special_tokens_are_parseable(self) -> None:
        adapter = self.adapter("R0")
        values = adapter(
            ["<think>计划</think>\npass<|im_end|>"],
            [["keywords:existence"]],
            [[{}]],
            ["raw-token"],
            completion_ids=[[1, 2, 3, 4]],
        )
        self.assertEqual(values, [1.0])
        self.assertEqual(adapter.records[0].completion_tokens, 4)

    def test_structured_answer_without_reasoning_keeps_reward_and_is_flagged(self) -> None:
        adapter = self.adapter("R0")
        values = adapter(
            [[{"role": "assistant", "content": "pass"}]],
            [["keywords:existence"]], [[{}]], ["answer-only"],
        )
        self.assertEqual(values, [1.0])
        self.assertTrue(adapter.records[0].reasoning_missing)

    def test_plain_string_answer_uses_explicit_fallback(self) -> None:
        adapter = self.adapter("R0")
        values = adapter(
            ["pass"], [["keywords:existence"]], [[{}]], ["plain"],
        )
        self.assertEqual(values, [1.0])
        self.assertTrue(adapter.records[0].answer_only_fallback)

    def test_arrow_none_fields_are_removed_before_checker(self) -> None:
        class StrictInstruction(FakeInstruction):
            def build_description(self, required=None, **kwargs):
                if required != "ok" or kwargs:
                    raise TypeError("unexpected Arrow padding fields")

        class StrictIfeval:
            INSTRUCTION_DICT = {"strict:test": StrictInstruction}

        adapter = RewardBatchAdapter(
            RewardCalculator(StrictIfeval()),
            load_algorithm_contract(CONFIG, "R0"),
            token_counter=lambda _: 1,
            maximum_completion_tokens=10,
            overlong_buffer_tokens=2,
        )
        values = adapter(
            ["pass"], [["strict:test"]],
            [[{"required": "ok", "unrelated": None}]], ["arrow"],
        )
        self.assertEqual(values, [1.0])
        self.assertEqual(adapter.calculator.checker_errors, 0)

    def test_r0_scores_only_answer_without_overlong_penalty(self) -> None:
        adapter = self.adapter("R0", token_count=10)
        result = adapter(
            ["secret fail reasoning</think>final pass"],
            [["keywords:existence"]],
            [[{}]],
            ["row-1"],
        )
        self.assertEqual(result, [1.0])
        self.assertEqual(adapter.records[0].overlong_penalty, 0.0)
        self.assertTrue(adapter.records[0].parse_ok)

    def test_r1_adds_soft_overlong_penalty(self) -> None:
        adapter = self.adapter("R1", token_count=9)
        result = adapter(
            ["reasoning</think>pass"],
            [["keywords:existence"]],
            [[{}]],
            ["row-1"],
        )
        self.assertEqual(result, [0.5])
        self.assertEqual(adapter.records[0].overlong_penalty, -0.5)

    def test_category_rates_and_record_drain(self) -> None:
        adapter = self.adapter("R2")
        adapter(
            ["reasoning</think>pass", "reasoning</think>failed"],
            [["keywords:existence"], ["keywords:existence"]],
            [[{}], [{}]],
            ["row-1", "row-2"],
        )
        self.assertEqual(adapter.category_pass_rates(), {"keywords": 0.5})
        records = adapter.pop_records()
        self.assertEqual(len(records), 2)
        self.assertEqual(adapter.records, [])

    def test_misaligned_trl_columns_fail_loudly(self) -> None:
        adapter = self.adapter("R0")
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            adapter(
                ["reasoning</think>pass"],
                [["keywords:existence"]],
                [[{}]],
                [],
            )


class RunPreflightTest(unittest.TestCase):
    def test_output_state_requires_explicit_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            (output / "checkpoint-50").mkdir()
            with self.assertRaisesRegex(ValueError, "use --resume"):
                validate_output_state(output, resume=False)
            self.assertEqual(validate_output_state(output, resume=True), 50)

    def test_build_preflight_supports_explicit_adapter_and_never_test_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_path = root / "train.jsonl"
            train_path.write_text(
                json.dumps(make_row(), ensure_ascii=False) + "\n", encoding="utf-8"
            )
            adapter = root / "T1" / "final_adapter"
            adapter.mkdir(parents=True)
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
            config = copy.deepcopy(CONFIG)
            config["data"]["train_size"] = 1
            config["rlvr"]["train_input"] = str(train_path)
            config["rlvr"]["initial_adapter"] = str(adapter)
            config["rlvr"]["initialization"] = "existing_adapter"
            config["rlvr"]["blocked_until_t1_v2_generation_gate"] = False
            config["rlvr"]["output_root"] = str(root / "outputs")
            config["rlvr"]["smoke_prompt_ids"] = ["rlvr-test"]
            manifest, rows = build_preflight(config, "R2", smoke=True, resume=False)
            self.assertEqual(manifest["requested_steps"], 5)
            self.assertEqual(manifest["zero_variance_abort_patience"], 8)
            self.assertEqual(manifest["train_rows"], 1)
            self.assertFalse(manifest["untouched_test_used"])
            self.assertEqual(manifest["initialization"], "existing_adapter")
            self.assertEqual(manifest["initial_adapter"], str(adapter))
            self.assertEqual(rows[0]["row_id"], "rlvr-test")

    def test_smoke_selection_is_separate_from_formal_pool(self) -> None:
        config = copy.deepcopy(CONFIG)
        config["rlvr"]["blocked_until_t1_v2_generation_gate"] = False
        config["rlvr"]["initialization"] = "fresh_lora_on_qwen3_4b"
        config["rlvr"]["initial_adapter"] = None
        smoke_manifest, smoke_rows = build_preflight(config, "R0", smoke=True, resume=False)
        formal_manifest, formal_rows = build_preflight(config, "R0", smoke=False, resume=False)
        self.assertEqual(smoke_manifest["train_rows"], 2000)
        self.assertEqual(smoke_manifest["active_train_rows"], 8)
        self.assertEqual(len(smoke_rows), 8)
        self.assertEqual(formal_manifest["active_train_rows"], 2000)
        self.assertEqual(len(formal_rows), 2000)
        self.assertEqual(smoke_manifest["smoke_prompt_ids"][:2], ["rlvr-0062", "rlvr-0502"])
        self.assertEqual(smoke_manifest["initialization"], "fresh_lora_on_qwen3_4b")
        self.assertIsNone(smoke_manifest["initial_adapter"])

    def test_rlvr_is_blocked_until_repaired_t1_passes_generation(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "T1_v2 free-generation"):
            build_preflight(CONFIG, "R0", smoke=True, resume=False)

    def test_initialization_mode_and_adapter_must_agree(self) -> None:
        config = copy.deepcopy(CONFIG)
        config["rlvr"]["initialization"] = "existing_adapter"
        config["rlvr"]["initial_adapter"] = None
        with self.assertRaisesRegex(ValueError, "requires rlvr.initial_adapter"):
            validate_rlvr_config(config)

        config = copy.deepcopy(CONFIG)
        config["rlvr"]["initialization"] = "fresh_lora_on_qwen3_4b"
        with self.assertRaisesRegex(ValueError, "requires rlvr.initial_adapter=null"):
            validate_rlvr_config(config)


class TrainerBindingTest(unittest.TestCase):
    def test_reward_decode_preserves_structural_special_tokens(self) -> None:
        class FakeTokenizer:
            def batch_decode(self, completion_ids, skip_special_tokens):
                self.call = (completion_ids, skip_special_tokens)
                return ["<think>reason</think>answer<|im_end|>"]

        tokenizer = FakeTokenizer()
        structured = [[{"role": "assistant", "content": "reasonanswer"}]]
        result = decode_reward_completions(tokenizer, structured, [[1, 2, 3]])
        self.assertEqual(result, ["<think>reason</think>answer<|im_end|>"])
        self.assertEqual(tokenizer.call, ([[1, 2, 3]], False))

    def test_smoke_health_requires_structure_variance_termination_and_update(self) -> None:
        healthy = [{
            "checker_errors": 0,
            "zero_variance_group_ratio": 0.5,
            "parse_ok_ratio": 1.0,
            "completion_at_limit_ratio": 0.5,
        }]
        summary = evaluate_smoke_health(healthy, "before", "after")
        self.assertTrue(summary["trainable_parameters_changed"])

        broken_cases = (
            ({**healthy[0], "checker_errors": 1}, "checker"),
            ({**healthy[0], "zero_variance_group_ratio": 1.0}, "zero reward variance"),
            ({**healthy[0], "parse_ok_ratio": 0.875}, "thinking/answer structure"),
            ({**healthy[0], "completion_at_limit_ratio": 1.0}, "generation limit"),
        )
        for stats, message in broken_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RuntimeError, message):
                    evaluate_smoke_health([stats], "before", "after")
        with self.assertRaisesRegex(RuntimeError, "did not change"):
            evaluate_smoke_health(healthy, "same", "same")
    def test_qwen_chat_generation_stops_at_im_end(self) -> None:
        class FakeTokenizer:
            token_ids = {"<|endoftext|>": 151643, "<|im_end|>": 151645}
            unk_token_id = None

            def __init__(self) -> None:
                self._pad_token = None
                self._eos_token = "<|endoftext|>"

            def convert_tokens_to_ids(self, token):
                return self.token_ids.get(token)

            @property
            def pad_token(self):
                return self._pad_token

            @pad_token.setter
            def pad_token(self, token):
                self._pad_token = token

            @property
            def pad_token_id(self):
                return self.token_ids.get(self._pad_token)

            @property
            def eos_token(self):
                return self._eos_token

            @eos_token.setter
            def eos_token(self, token):
                self._eos_token = token

            @property
            def eos_token_id(self):
                return self.token_ids.get(self._eos_token)

        tokenizer = FakeTokenizer()
        state = configure_qwen_chat_termination(tokenizer)
        self.assertEqual(state["eos_token_id"], 151645)
        self.assertEqual(state["pad_token_id"], 151643)

    def test_learning_signal_detection(self) -> None:
        self.assertFalse(advantages_have_learning_signal([0.0, 0.0, 0.0], 1e-8))
        self.assertFalse(advantages_have_learning_signal([1e-10, -1e-10], 1e-8))
        self.assertTrue(advantages_have_learning_signal([0.0, 0.25, -0.25], 1e-8))

    def test_r0_r1_r2_grpo_mapping_preserves_attribution(self) -> None:
        r0 = build_grpo_kwargs(CONFIG, "R0", Path("r0"), False, True)
        r1 = build_grpo_kwargs(CONFIG, "R1", Path("r1"), False, True)
        r2 = build_grpo_kwargs(CONFIG, "R2", Path("r2"), False, True)
        self.assertEqual(r0["loss_type"], "grpo")
        self.assertEqual(r0["epsilon_high"], 0.2)
        self.assertFalse(r0["mask_truncated_completions"])
        self.assertEqual(r1["loss_type"], "dapo")
        self.assertEqual(r1["epsilon_high"], 0.28)
        self.assertTrue(r1["mask_truncated_completions"])
        # R1/R2 的 TRL 参数必须完全一致，唯一差异只能在 trainer sampler。
        self.assertEqual({k: v for k, v in r1.items() if k != "output_dir"}, {k: v for k, v in r2.items() if k != "output_dir"})

    def test_weighted_sampler_keeps_four_completions_contiguous(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is unavailable")
        sampler = WeightedRepeatSampler(
            list(range(8)), mini_repeat_count=4, batch_size=2,
            repeat_count=1, weights_fn=lambda: [1.0] * 8, seed=42,
        )
        values = list(sampler)
        self.assertEqual(len(values), 32)
        self.assertEqual(len(sampler), 32)
        for start in range(0, len(values), 8):
            self.assertEqual(len(set(values[start:start + 4])), 1)
            self.assertEqual(len(set(values[start + 4:start + 8])), 1)
            self.assertNotEqual(values[start], values[start + 4])


if __name__ == "__main__":
    unittest.main()
