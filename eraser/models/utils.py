"""Summary prompts shared by model inference and identifier mapping."""


def _build_prompt(code, tokenizer=None, lang="python"):
    return (
        "Provide a concise English summary (<= 100 word) of the main purpose.\n\n"
        f"```{lang}\n{code}\n```\nSummary:"
    )


def _build_prompt_w_chat(code, tokenizer, lang="python"):
    content = (
        "Provide a concise English summary (<= 100 word) of the main purpose.\n"
        f"```{lang}\n{code}\n```\nSummary:"
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True,
    )
