"""Qwen3 evaluation generation IDs shared by vLLM entry points."""


def qwen3_generation_ids(tokenizer):
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if not isinstance(im_end_id, int) or im_end_id < 0:
        raise RuntimeError("Tokenizer does not contain <|im_end|>")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    return im_end_id, pad_id
