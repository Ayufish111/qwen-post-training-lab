from __future__ import annotations

from collections.abc import Mapping, Sequence


VALID_ROLES = {"system", "user", "assistant", "tool"}


def validate_messages(messages: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        return ["messages must be a list"]
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            errors.append(f"messages[{index}] must be an object")
            continue
        role = message.get("role")
        if role not in VALID_ROLES:
            errors.append(f"messages[{index}].role is invalid: {role!r}")
        if "content" not in message and "tool_calls" not in message:
            errors.append(f"messages[{index}] needs content or tool_calls")
        if "content" in message and message["content"] is not None and not isinstance(message["content"], str):
            errors.append(f"messages[{index}].content must be a string or null")
    return errors


def validate_sft_record(record: object) -> list[str]:
    if not isinstance(record, Mapping):
        return ["record must be an object"]
    errors = validate_messages(record.get("messages"))
    messages = record.get("messages")
    if isinstance(messages, Sequence) and messages:
        last_message = messages[-1]
        if isinstance(last_message, Mapping) and last_message.get("role") != "assistant":
            errors.append("the final SFT message must be assistant")
    return errors


def validate_dpo_record(record: object) -> list[str]:
    if not isinstance(record, Mapping):
        return ["record must be an object"]
    errors: list[str] = []
    for field in ("prompt", "chosen", "rejected"):
        if field not in record:
            errors.append(f"missing field: {field}")
        else:
            errors.extend(f"{field}: {error}" for error in validate_messages(record[field]))
    return errors
