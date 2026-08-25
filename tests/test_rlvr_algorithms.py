"""CPU-only unit tests for RLVR reward and CA-DAPO sampler formulas.

Covers plan section 6.1 (rewards), 6.2 (GRPO), 6.3 (DAPO-style mechanisms)
and 6.4 (Constraint-Aware sampling). No torch/transformers and no
official Multi-IF import: a small fake checker replaces the official module.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src import constraint_sampler as sampler  # noqa: E402
from src import rlvr_rewards as rewards  # noqa: E402


class FakeInstruction:
    """Minimal checker: instruction id prefixes decide whether it passes."""

    def __init__(self, instruction_id: str) -> None:
        self.id = instruction_id

    def build_description(self, **kwargs) -> None:
        self.kwargs = kwargs

    def check_following(self, value) -> bool:
        return self.id.startswith("pass")


class FakeIeval:
    INSTRUCTION_DICT = {
        "pass_keywords:existence": FakeInstruction,
        "pass_detectable_format:title": FakeInstruction,
        "fail_keywords:forbidden_words": FakeInstruction,
        "fail_startend:end_checker": FakeInstruction,
    }


def category_of(instruction_id: str) -> str:
    """Real RLVR rows store the pre-colon prefix (e.g. ``keywords``).

    The test checker prefixes ids with ``pass_``/``fail_`` only to encode the
    expected result, so this helper strips that marker before using the id as
    a real constraint category.
    """
    raw = instruction_id.split(":", maxsplit=1)[0]
    if raw.startswith(("pass_", "fail_")):
        return raw.split("_", maxsplit=1)[1]
    return raw


def make_row(
    instruction_ids: list[str],
    kwargs: list[dict] | None = None,
    row_id: str = "test-row",
) -> dict:
    return {
        "id": row_id,
        "instruction_ids": instruction_ids,
        "kwargs": kwargs if kwargs is not None else [{} for _ in instruction_ids],
        "constraint_categories": [
            category_of(value) for value in instruction_ids
        ],
        "metadata": {},
    }


class ParseReasoningAnswerTest(unittest.TestCase):
    def test_accepts_implicit_opening_marker(self) -> None:
        self.assertEqual(
            rewards.parse_reasoning_answer("先列约束。\n</think>\n最终答案"),
            ("先列约束。", "最终答案"),
        )

    def test_accepts_explicit_opening_and_strips_end_markers(self) -> None:
        self.assertEqual(
            rewards.parse_reasoning_answer(
                "<think>step one</think>answer<|im_end|><|endoftext|>"
            ),
            ("step one", "answer"),
        )

    def test_rejects_missing_or_empty_sections_and_non_string(self) -> None:
        self.assertIsNone(rewards.parse_reasoning_answer("answer only"))
        self.assertIsNone(rewards.parse_reasoning_answer("reason</think>"))
        self.assertIsNone(rewards.parse_reasoning_answer("</think>answer"))
        self.assertIsNone(rewards.parse_reasoning_answer(None))


class RewardCalculatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.calculator = rewards.RewardCalculator(FakeIeval())

    def test_score_formula_all_pass(self) -> None:
        score = rewards.RewardCalculator.score([True, True])
        self.assertEqual(score.r_instruction, 1.0)
        self.assertEqual(score.r_prompt, 1.0)
        self.assertAlmostEqual(score.r_core, 0.7 * 1.0 + 0.3 * 1.0)
        self.assertEqual(score.instruction_count, 2)
        self.assertEqual(score.passed_count, 2)

    def test_score_formula_partial_pass(self) -> None:
        score = rewards.RewardCalculator.score([True, False])
        self.assertAlmostEqual(score.r_instruction, 0.5)
        self.assertEqual(score.r_prompt, 0.0)
        self.assertAlmostEqual(score.r_core, 0.7 * 0.5 + 0.3 * 0.0)

    def test_score_formula_all_fail(self) -> None:
        score = rewards.RewardCalculator.score([False, False])
        self.assertAlmostEqual(score.r_instruction, 0.0)
        self.assertEqual(score.r_prompt, 0.0)
        self.assertEqual(score.r_core, 0.0)

    def test_empty_answer_scores_false_for_every_constraint(self) -> None:
        results = self.calculator.check_constraints(
            "   ",
            ["pass_keywords:existence", "fail_keywords:forbidden_words"],
            [{}, {}],
        )
        self.assertEqual(results, [False, False])

    def test_checker_metadata_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.calculator.check_constraints(
                "answer",
                ["pass_keywords:existence", "pass_detectable_format:title"],
                [{}],
            )

    def test_reward_scores_final_answer_only(self) -> None:
        row = make_row(["pass_keywords:existence"])
        score, parsed = self.calculator.reward(
            row, "<think>计划</think>\n\n满足要求的答案"
        )
        self.assertEqual(parsed, ("计划", "满足要求的答案"))
        self.assertEqual(score.r_prompt, 1.0)

    def test_reward_unparsable_output_scores_zero(self) -> None:
        row = make_row(["pass_keywords:existence"])
        score, parsed = self.calculator.reward(row, "没有思考标记的回答")
        self.assertIsNone(parsed)
        self.assertEqual(score.r_instruction, 0.0)
        self.assertEqual(score.r_prompt, 0.0)
        self.assertEqual(score.r_core, 0.0)

    def test_checker_exception_becomes_failure_instead_of_crashing(self) -> None:
        class BrokenInstruction(FakeInstruction):
            def check_following(self, _: str) -> bool:
                raise RuntimeError("bad checker parameter")

        class BrokenIfeval:
            INSTRUCTION_DICT = {"broken:constraint": BrokenInstruction}

        calculator = rewards.RewardCalculator(BrokenIfeval())
        results = calculator.check_constraints("answer", ["broken:constraint"], [{}])
        self.assertEqual(results, [False])
        self.assertEqual(calculator.checker_errors, 1)


class GrpoAdvantageTest(unittest.TestCase):
    def test_standard_sample_std_advantage(self) -> None:
        advantages = rewards.group_advantages([1.0, 2.0, 3.0])
        # Sample std of [1,2,3] is 1.0. TRL divides by std + 1e-4.
        expected_first = -1.0 / 1.0001
        self.assertAlmostEqual(advantages[0], expected_first)
        self.assertAlmostEqual(advantages[2], -expected_first)
        # Advantages are mean-centered across the group.
        self.assertAlmostEqual(sum(advantages), 0.0, places=12)

    def test_empty_input(self) -> None:
        self.assertEqual(rewards.group_advantages([]), [])

    def test_single_reward_has_zero_advantage(self) -> None:
        self.assertEqual(rewards.group_advantages([0.5]), [0.0])


class DapoClipHigherTest(unittest.TestCase):
    def test_clip_within_interval_is_unclipped_value(self) -> None:
        value = rewards.clip_higher(1.0, 1.0, 0.2, 0.28)
        self.assertAlmostEqual(value, -1.0)

    def test_upper_clip_higher_differs_from_symmetric_grpo(self) -> None:
        # ratio=1.25 is inside the DAPO upper bound (1.28) but would be
        # clipped by symmetric GRPO (1.2). With positive advantage the clipped
        # term is larger, so min() picks the unclipped value.
        value = rewards.clip_higher(1.25, 1.0, 0.2, 0.28)
        self.assertAlmostEqual(value, -1.25)

    def test_lower_clip_applies_for_negative_advantage(self) -> None:
        # ratio=0.5 is below the lower bound 0.8. With negative advantage the
        # clipped term (-0.8) is smaller than the unclipped term (-0.5), so
        # min() selects the clipped objective (-0.8); negation produces the
        # minimization loss +0.8.
        value = rewards.clip_higher(0.5, -1.0, 0.2, 0.28)
        self.assertAlmostEqual(value, 0.8)


class DapoTokenLevelLossTest(unittest.TestCase):
    def test_matches_manual_average_over_valid_tokens(self) -> None:
        ratios = [1.0, 1.25]
        advantages = [1.0, 1.0]
        weights = [1.0, 1.0]
        expected = (
            rewards.clip_higher(1.0, 1.0, 0.2, 0.28)
            + rewards.clip_higher(1.25, 1.0, 0.2, 0.28)
        ) / 2
        self.assertAlmostEqual(
            rewards.token_level_policy_loss(
                ratios, advantages, weights, 0.2, 0.28
            ),
            expected,
        )

    def test_zero_weights_return_zero(self) -> None:
        self.assertEqual(
            rewards.token_level_policy_loss(
                [1.0, 1.25], [1.0, 1.0], [0.0, 0.0], 0.2, 0.28
            ),
            0.0,
        )

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            rewards.token_level_policy_loss(
                [1.0, 1.0], [1.0], [1.0, 1.0], 0.2, 0.28
            )


class DapoDynamicSamplingTest(unittest.TestCase):
    def test_identical_rewards_need_resampling(self) -> None:
        self.assertTrue(rewards.group_has_zero_reward_variance([0.5, 0.5, 0.5]))

    def test_varying_rewards_are_kept(self) -> None:
        self.assertFalse(rewards.group_has_zero_reward_variance([0.2, 0.5, 0.9]))

    def test_empty_and_single_groups_need_resampling(self) -> None:
        self.assertTrue(rewards.group_has_zero_reward_variance([]))
        self.assertTrue(rewards.group_has_zero_reward_variance([1.0]))


class DapoSoftOverlongTest(unittest.TestCase):
    def test_threshold_is_max_length_minus_buffer(self) -> None:
        self.assertEqual(rewards.soft_overlong_threshold(512, 64), 448)

    def test_short_completion_no_penalty(self) -> None:
        self.assertEqual(rewards.soft_overlong_penalty(200, 512, 64), 0.0)

    def test_linear_penalty_in_buffer_zone(self) -> None:
        # 480 is 32 tokens past the 448 threshold within a 64-token buffer.
        self.assertAlmostEqual(rewards.soft_overlong_penalty(480, 512, 64), -0.5)

    def test_penalty_reaches_zero_at_hard_limit(self) -> None:
        self.assertEqual(rewards.soft_overlong_penalty(512, 512, 64), -1.0)
        self.assertEqual(rewards.soft_overlong_penalty(600, 512, 64), -1.0)


class ConstraintSamplerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sampler = sampler.ConstraintSampler()

    def test_learning_frontier_peaks_at_half_pass_rate(self) -> None:
        self.assertAlmostEqual(sampler.category_learnability(0.0), 0.0)
        self.assertAlmostEqual(sampler.category_learnability(0.5), 1.0)
        self.assertAlmostEqual(sampler.category_learnability(1.0), 0.0)
        self.assertEqual(sampler.category_progress_signal(0.5, 0.5), 0.0)
        self.assertGreater(sampler.category_progress_signal(0.5, 0.7), 0.0)

    def test_initial_weights_are_uniform(self) -> None:
        rows = [
            make_row(["pass_keywords:existence"]),
            make_row(["fail_keywords:forbidden_words"]),
        ]
        weights = self.sampler.sample_weights(rows)
        self.assertEqual(weights, [1.0, 1.0])

    def test_update_cadence_is_20_steps(self) -> None:
        sampler20 = sampler.ConstraintSampler(update_steps=20)
        self.assertFalse(sampler20.should_update(0))
        self.assertFalse(sampler20.should_update(19))
        self.assertTrue(sampler20.should_update(20))
        self.assertTrue(sampler20.should_update(40))

    def test_production_weights_change_only_at_update_step(self) -> None:
        inst = sampler.ConstraintSampler(update_steps=20)
        rows = [
            make_row(["fail_keywords:existence"], row_id="hard"),
            make_row(["pass_startend:end_checker"], row_id="easy"),
        ]
        self.assertEqual(inst.weights_for_step(rows, 0), [1.0, 1.0])
        inst.update_batch_pass_rates({"keywords": 0.2, "startend": 0.0})
        inst.update_batch_pass_rates({"keywords": 0.5, "startend": 0.0})
        self.assertEqual(inst.weights_for_step(rows, 19), [1.0, 1.0])
        self.assertEqual(inst.weights_for_step(rows, 20), [1.0, 0.5])
        inst.update_batch_pass_rates({"keywords": 0.5, "startend": 0.0})
        self.assertEqual(inst.weights_for_step(rows, 21), [1.0, 0.5])

    def test_production_weights_reject_pool_reordering(self) -> None:
        inst = sampler.ConstraintSampler()
        rows = [make_row([], row_id="a"), make_row([], row_id="b")]
        inst.weights_for_step(rows, 0)
        with self.assertRaises(ValueError):
            inst.weights_for_step(list(reversed(rows)), 1)

    def test_ema_update_formula(self) -> None:
        inst = sampler.ConstraintSampler(ema_beta=0.9)
        inst.update_batch_pass_rates({"detectable_format": 0.5})
        self.assertAlmostEqual(inst.category_pass_ema["detectable_format"], 0.5)
        inst.update_batch_pass_rates({"detectable_format": 1.0})
        self.assertAlmostEqual(
            inst.category_pass_ema["detectable_format"], 0.9 * 0.5 + 0.1 * 1.0
        )

    def test_invalid_batch_pass_rate_raises(self) -> None:
        inst = sampler.ConstraintSampler()
        for value in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                inst.update_batch_pass_rates({"keywords": value})

    def test_sampler_state_round_trip_preserves_active_weights(self) -> None:
        rows = [
            make_row(["fail_keywords:existence"], row_id="hard"),
            make_row(["pass_startend:end_checker"], row_id="easy"),
        ]
        original = sampler.ConstraintSampler()
        original.weights_for_step(rows, 0)
        original.update_batch_pass_rates({"keywords": 0.2, "startend": 1.0})
        original.update_batch_pass_rates({"keywords": 0.5, "startend": 1.0})
        expected = original.weights_for_step(rows, 20)

        restored = sampler.ConstraintSampler()
        restored.load_state_dict(original.state_dict())
        self.assertEqual(restored.weights_for_step(rows, 21), expected)
        self.assertEqual(restored.category_pass_ema, original.category_pass_ema)
        self.assertEqual(restored.last_weight_update_step, 20)

    def test_learnability_unknown_category_returns_none(self) -> None:
        self.assertIsNone(self.sampler.learnability("never_seen"))

    def test_prompt_learnability_ignores_unseen_categories(self) -> None:
        inst = sampler.ConstraintSampler()
        inst.update_batch_pass_rates(
            {"detectable_format": 0.9, "keywords": 0.1}
        )
        inst.update_batch_pass_rates(
            {"detectable_format": 0.9, "keywords": 0.3}
        )
        row = make_row(["fail_keywords:existence", "pass_detectable_format:title"])
        self.assertGreater(inst.prompt_learnability(row), 0.0)
        # A row with no observed categories keeps the uniform default.
        self.assertIsNone(inst.prompt_learnability(make_row([])))
        self.assertIsNone(
            inst.prompt_learnability(make_row(["pass_startend:end_checker"]))
        )

    def test_learning_frontier_category_gets_higher_weight_after_update(self) -> None:
        inst = sampler.ConstraintSampler()
        # A category at the learning frontier (50% pass) outranks an extreme.
        inst.update_batch_pass_rates({"keywords": 0.2, "startend": 0.0})
        inst.update_batch_pass_rates({"keywords": 0.5, "startend": 0.0})
        rows = [
            make_row(["fail_keywords:existence"]),
            make_row(["pass_startend:end_checker"]),
        ]
        weights = inst.sample_weights(rows)
        # keywords learnability 1.0 -> normalized 1.0 -> weight
        #   (1-0.5) + 0.5*1.0 = 1.0
        # startend learnability 0.0 -> normalized 0.0 -> weight 0.5
        self.assertAlmostEqual(weights[0], 1.0)
        self.assertAlmostEqual(weights[1], 0.5)
        # The hard prompt is weighted twice as much as the easy one.
        self.assertAlmostEqual(weights[0], 2.0 * weights[1])

    def test_unseen_prompt_keeps_uniform_default_weight(self) -> None:
        inst = sampler.ConstraintSampler()
        inst.update_batch_pass_rates({"keywords": 0.2})
        inst.update_batch_pass_rates({"keywords": 0.5})
        rows = [
            make_row(["fail_keywords:existence"]),
            make_row(["pass_startend:end_checker"]),  # startend unseen
        ]
        weights = inst.sample_weights(rows)
        # Only one learnability value is known, so min-max normalization maps the
        # single value to 0.5; weight = 0.5 + 0.5*0.5 = 0.75.
        self.assertAlmostEqual(weights[0], 0.75)
        # The unseen prompt keeps the uniform default weight 1.0.
        self.assertEqual(weights[1], 1.0)

    def test_weights_respect_frozen_clip_bounds(self) -> None:
        inst = sampler.ConstraintSampler(
            mixture_lambda=0.5, min_sampling_weight=0.5, max_sampling_weight=2.0
        )
        rows = [
            make_row(["pass_startend:end_checker"]),
            make_row(["fail_keywords:existence"]),
        ]
        inst.update_batch_pass_rates({"startend": 0.0, "keywords": 0.2})
        inst.update_batch_pass_rates({"startend": 0.0, "keywords": 0.5})
        weights = inst.sample_weights(rows)
        # Frozen formula range for lambda=0.5 is [0.5, 1.0]; clip bounds are a
        # safety cage, so the returned weights stay inside [0.5, 2.0].
        self.assertAlmostEqual(min(weights), 0.5)
        self.assertAlmostEqual(max(weights), 1.0)
        self.assertTrue(all(0.5 <= value <= 2.0 for value in weights))

    def test_stagnation_scale_reduces_transient_progress_signal(self) -> None:
        inst = sampler.ConstraintSampler()
        inst.update_batch_pass_rates({"keywords": 0.2})
        inst.update_batch_pass_rates({"keywords": 0.5})
        before = inst.learnability("keywords")
        for _ in range(3):
            inst.update_batch_pass_rates({"keywords": 0.5})
        after = inst.learnability("keywords")
        self.assertIsNotNone(before)
        self.assertIsNotNone(after)
        self.assertLess(after, before)
        self.assertEqual(inst.category_stagnant_updates["keywords"], 3)

    def test_normalize_equal_values_maps_to_half(self) -> None:
        self.assertEqual(sampler.normalize_learnabilities([0.3, 0.3]), [0.5, 0.5])
        self.assertEqual(sampler.normalize_learnabilities([]), [])


if __name__ == "__main__":
    unittest.main()
