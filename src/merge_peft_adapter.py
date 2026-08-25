"""Merge a PEFT LoRA adapter, including trainable token deltas, into a base model."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_config_path = args.adapter / "adapter_config.json"
    adapter_weights_path = args.adapter / "adapter_model.safetensors"
    if not adapter_config_path.is_file() or not adapter_weights_path.is_file():
        raise FileNotFoundError(f"Incomplete adapter: {args.adapter}")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"Output directory is not empty: {args.output}")

    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    token_ids = [int(value) for value in adapter_config.get("trainable_token_indices") or []]
    if not token_ids:
        raise ValueError("Adapter does not declare trainable_token_indices")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )
    embeddings = model.get_input_embeddings().weight.detach()
    if max(token_ids) >= embeddings.shape[0]:
        raise ValueError(
            f"Token id {max(token_ids)} exceeds embedding rows {embeddings.shape[0]}"
        )
    before = embeddings[token_ids].float().cpu().clone()

    peft_model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
    merged = peft_model.merge_and_unload(safe_merge=True)
    after = merged.get_input_embeddings().weight.detach()[token_ids].float().cpu()
    changes = (after - before).abs().amax(dim=1).tolist()
    if not any(value > 0 for value in changes):
        raise RuntimeError("Trainable token rows did not change during adapter merge")

    remaining_wrappers = [
        name
        for name, module in merged.named_modules()
        if "lora" in module.__class__.__name__.lower()
        or "trainabletokens" in module.__class__.__name__.lower()
    ]
    if remaining_wrappers:
        raise RuntimeError(f"PEFT wrappers remain after merge: {remaining_wrappers[:5]}")

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if not isinstance(im_end_id, int) or im_end_id < 0:
        raise RuntimeError("Tokenizer does not contain <|im_end|>")
    merged.config.eos_token_id = im_end_id
    merged.config.pad_token_id = pad_id
    if getattr(merged, "generation_config", None) is not None:
        merged.generation_config.eos_token_id = im_end_id
        merged.generation_config.pad_token_id = pad_id

    args.output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(
        args.output, safe_serialization=True, max_shard_size="4GB"
    )
    tokenizer_artifacts = (
        "added_tokens.json",
        "chat_template.jinja",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
    )
    copied_tokenizer_artifacts = []
    for filename in tokenizer_artifacts:
        source = args.model / filename
        if source.is_file():
            shutil.copy2(source, args.output / filename)
            copied_tokenizer_artifacts.append(filename)
    if "tokenizer_config.json" not in copied_tokenizer_artifacts:
        raise FileNotFoundError(f"Base tokenizer artifacts are incomplete: {args.model}")
    manifest = {
        "schema_version": "merged_peft_model/v1",
        "base_model": str(args.model.resolve()),
        "adapter": str(args.adapter.resolve()),
        "adapter_sha256": sha256_file(adapter_weights_path),
        "adapter_peft_version": adapter_config.get("peft_version"),
        "trainable_token_indices": token_ids,
        "trainable_token_max_abs_changes": changes,
        "eos_token_id": im_end_id,
        "pad_token_id": pad_id,
        "dtype": str(next(merged.parameters()).dtype),
        "safe_merge": True,
        "tokenizer_artifacts": copied_tokenizer_artifacts,
    }
    (args.output / "merge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"merged model saved to: {args.output}")


if __name__ == "__main__":
    main()