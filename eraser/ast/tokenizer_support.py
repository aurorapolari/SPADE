"""Prompt-token offsets without changing the tokenizer's token indices."""
from dataclasses import dataclass


@dataclass
class EncodedCode:
    inputs: dict
    occurrences: list
    identifiers: list
    context: list


def get_tokenizer_map(tokenizer, code, *, prompt_proc=None, strict=True, ref_func=None):
    if prompt_proc is None:
        from ..models.utils import _build_prompt
        prompt_proc = _build_prompt
    prompt = prompt_proc(code, ref_func, tokenizer) if ref_func is not None else prompt_proc(code, tokenizer)
    start = prompt.index(code)
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    mapping = [
        {"id": int(token_id), "token": prompt[left:right], "offset_start": left, "offset_end": right}
        for token_id, (left, right) in zip(encoded["input_ids"], encoded["offset_mapping"])
    ]
    return mapping, start, start + len(code)
