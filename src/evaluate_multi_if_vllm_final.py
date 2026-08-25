"""Frozen vLLM entry point with explicit Qwen3 <|im_end|> stopping."""

import importlib

import evaluate_multi_if_vllm as base
from vllm_eos_config import qwen3_generation_ids


def load_vllm(model_path, adapter_path, tensor_parallel_size, gpu_memory_utilization):
    try:
        vllm = importlib.import_module("vllm")
    except ImportError as exc:
        raise RuntimeError("vLLM is not installed; run this on AutoDL/Linux.") from exc
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    qwen3_generation_ids(tokenizer)
    llm_kwargs = {
        "model": model_path,
        "enable_lora": bool(adapter_path),
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": False,
    }
    if adapter_path:
        llm_kwargs["max_lora_rank"] = 64
    llm = vllm.LLM(**llm_kwargs)
    request = None
    if adapter_path:
        from vllm.lora.request import LoRARequest
        request = LoRARequest("evaluation_adapter", 1, str(adapter_path))
    return vllm, tokenizer, llm, request


def generate_one(vllm, tokenizer, llm, request, messages, max_tokens, enable_thinking):
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    im_end_id, _pad_id = qwen3_generation_ids(tokenizer)
    sampling = vllm.SamplingParams(
        temperature=0.0,
        top_p=1.0,
        skip_special_tokens=False,
        max_tokens=max_tokens,
        stop_token_ids=[im_end_id],
    )
    import time
    started = time.perf_counter()
    generate_kwargs = {"sampling_params": sampling}
    if request is not None:
        generate_kwargs["lora_request"] = request
    completion = llm.generate([prompt], **generate_kwargs)[0].outputs[0]
    elapsed = time.perf_counter() - started
    token_ids = getattr(completion, "token_ids", ()) or ()
    raw_text = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    final_content = base.extract_final_content(
        raw_text, enable_thinking=enable_thinking
    )
    finish_reason = getattr(completion, "finish_reason", None)
    saw_thinking_tokens = "<think>" in raw_text or "</think>" in raw_text
    return final_content, {
        "raw_text": raw_text,
        "generated_tokens": len(token_ids),
        "generation_seconds": elapsed,
        "natural_eos": finish_reason != "length",
        "clipped": finish_reason == "length",
        "finish_reason": finish_reason,
        "saw_thinking_tokens": saw_thinking_tokens,
        "thinking_structure_ok": base.thinking_structure_ok(
            raw_text, enable_thinking=enable_thinking
        ),
        "enable_thinking": enable_thinking,
    }


def main():
    base.load_vllm = load_vllm
    base.generate_one = generate_one
    base.main()


if __name__ == "__main__":
    main()
