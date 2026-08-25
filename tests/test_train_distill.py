from __future__ import annotations

import unittest

import torch

from src.train_distill import (
    inspect_generated_structure,
    lora_b_update_audit,
    resolve_structure_token_ids,
    select_smoke_rows,
    structure_weighted_causal_loss,
)


class FakeTokenizer:
    ids = {"<think>": 10, "</think>": 11, "<|im_end|>": 12}

    def encode(self, token, add_special_tokens=False):
        return [self.ids[token]]

    def convert_ids_to_tokens(self, token_id):
        return {value: key for key, value in self.ids.items()}[token_id]


class FakeDataset:
    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if key == "id":
            return [row["id"] for row in self.rows]
        return self.rows[key]

    def select(self, indexes):
        return FakeDataset([self.rows[index] for index in indexes])


class StructureTokenTest(unittest.TestCase):
    def test_resolves_exact_single_token_ids(self) -> None:
        self.assertEqual(
            resolve_structure_token_ids(
                FakeTokenizer(), ["<think>", "</think>", "<|im_end|>"]
            ),
            [10, 11, 12],
        )

    def test_structure_loss_uses_normalized_token_multiplier(self) -> None:
        logits = torch.zeros((1, 3, 4), dtype=torch.float32)
        logits[0, 1, 3] = 5.0
        labels = torch.tensor([[-100, 2, 3]])
        outputs = type("Outputs", (), {"loss": torch.tensor(0.5), "logits": logits})()
        combined = structure_weighted_causal_loss(outputs, labels, [2], 4.0)
        self.assertGreater(combined.item(), outputs.loss.item())

    def test_rejects_batch_without_supervised_tokens(self) -> None:
        logits = torch.zeros((1, 3, 4), dtype=torch.float32)
        labels = torch.full((1, 3), -100)
        outputs = type("Outputs", (), {"loss": torch.tensor(0.5), "logits": logits})()
        with self.assertRaisesRegex(RuntimeError, "no supervised structure"):
            structure_weighted_causal_loss(outputs, labels, [2], 4.0)

    def test_config_uses_structure_ids_without_trainable_token_wrapper(self) -> None:
        import yaml
        from pathlib import Path

        config = yaml.safe_load(
            (Path(__file__).resolve().parents[1] / "configs" / "rlvr.yaml").read_text(
                encoding="utf-8"
            )
        )
        qlora = config["t1"]["qlora"]
        self.assertEqual(
            qlora["structure_token_strings"],
            ["<think>", "</think>", "<|im_end|>"],
        )
        self.assertNotIn("trainable_token_strings", qlora)
        self.assertNotIn("lm_head", qlora["target_modules"])
        self.assertNotIn("lm_head", qlora["legacy_target_modules"])

    def test_adapter_update_audit_requires_structure_token_delta(self) -> None:
        model = torch.nn.Module()
        model.register_parameter("layer_lora_B", torch.nn.Parameter(torch.ones(2, 2)))
        model.register_parameter("trainable_tokens_delta", torch.nn.Parameter(torch.ones(3, 2)))
        audit = lora_b_update_audit(model)
        self.assertEqual(audit["nonzero_lora_b_tensor_count"], 1)
        self.assertTrue(audit["structure_token_delta_nonzero"])

    def test_smoke_selection_keeps_gate_row_and_rotates_pool(self) -> None:
        dataset = FakeDataset([{"id": f"row-{index}"} for index in range(6)])
        selected = select_smoke_rows(dataset, 3, 4, ["row-1", "missing"])
        self.assertEqual(selected["id"], ["row-1", "row-4", "row-5"])

    def test_strict_generation_requires_one_ordered_thinking_block(self) -> None:
        valid = inspect_generated_structure([10, 7, 11, 8, 12], 10, 11, 12)
        self.assertTrue(valid["strict_structure_ok"])
        repeated = inspect_generated_structure([10, 7, 10, 11, 12], 10, 11, 12)
        self.assertFalse(repeated["strict_structure_ok"])
        wrong_start = inspect_generated_structure([7, 10, 11, 12], 10, 11, 12)
        self.assertFalse(wrong_start["strict_structure_ok"])


if __name__ == "__main__":
    unittest.main()
