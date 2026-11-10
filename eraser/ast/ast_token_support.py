"""Identifier-to-token mapping with the original mapper interface."""
from .run_parser import identifier_spans
from .tokenizer_support import EncodedCode, get_tokenizer_map


class IdentiTokenMapper:
    def __init__(self, tokenizer, prompt_proc=None):
        self.tokenizer = tokenizer
        self.prompt_proc = prompt_proc

    def _map(self, code, func_name, lang, ref_func=None):
        mapping, code_start, code_end = get_tokenizer_map(
            self.tokenizer, code, prompt_proc=self.prompt_proc, ref_func=ref_func,
        )
        try:
            spans = identifier_spans(code, lang, func_name)
        except Exception:
            spans = []
        groups, identifiers, by_name, positions = [], set(), {}, {}
        for number, (name, start, end) in enumerate(spans):
            indexes = [i for i, token in enumerate(mapping)
                       if token["offset_end"] > token["offset_start"]
                       and token["offset_end"] > code_start + start
                       and token["offset_start"] < code_start + end]
            if not indexes:
                continue
            groups.append(indexes)
            identifiers.update(indexes)
            by_name.setdefault(name, []).append(indexes)
            line = code.count("\n", 0, start)
            line_start = code.rfind("\n", 0, start) + 1
            positions.setdefault(name, []).append([number, line, start - line_start, end - line_start])
        code_indexes = [i for i, token in enumerate(mapping)
                        if token["offset_end"] > token["offset_start"]
                        and token["offset_end"] > code_start and token["offset_start"] < code_end]
        return groups, sorted(identifiers), sorted(set(code_indexes) - identifiers), by_name, positions, code_indexes

    def one_identifier2token(self, code, func_name="", lang="python", debug=False, ref_func=None):
        try:
            _, identifiers, _, by_name, positions, code_indexes = self._map(code, func_name, lang, ref_func)
        except Exception:
            return None, None, None, None
        code_range = [code_indexes[0], code_indexes[-1]] if code_indexes else [None, None]
        return identifiers, by_name, positions, code_range

    def encode(self, code, func_name="", lang="python"):
        from ..models.utils import _build_prompt
        prompt_proc = self.prompt_proc or _build_prompt
        prompt = prompt_proc(code, self.tokenizer)
        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        groups, identifiers, context, *_ = self._map(code, func_name, lang)
        return EncodedCode(dict(inputs), groups, identifiers, context)
