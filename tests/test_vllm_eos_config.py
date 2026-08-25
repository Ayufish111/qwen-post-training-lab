import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vllm_eos_config import qwen3_generation_ids


class Tokenizer:
    eos_token_id = 151643
    pad_token_id = 151643

    def convert_tokens_to_ids(self, token):
        return {"<|im_end|>": 151645, "<|endoftext|>": 151643}.get(token, -1)


class VllmEosConfigTest(unittest.TestCase):
    def test_uses_im_end_not_default_eos(self):
        self.assertEqual(qwen3_generation_ids(Tokenizer()), (151645, 151643))
