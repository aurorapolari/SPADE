"""Native activation capture and selective erasure hooks."""
import json
from pathlib import Path

import torch

from .base import BaseHook
from .utils import map_prompt_token_indices_with_left_padding
from ..erasers.erasers import load_eraser_from_ckpt
from ..gate.features import gate_feature
from ..gate.gate import GateModel


class CaptureHook(BaseHook):
    def __init__(self, tk_lists=None, attention_mask=None, code_ids=None, capture_noindent=False, detach=True):
        super().__init__()
        self.tk_lists, self.attention_mask, self.code_ids = tk_lists, attention_mask, code_ids
        self.capture_noindent, self.detach = capture_noindent, detach
        self.captured = None
        self.no_indent_mean, self.no_indent_max = None, None
        self._done = False

    def _hook_fn(self, module, inputs, output):
        if self._done:
            return None
        self._done = True
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        self.captured, self.no_indent_mean, self.no_indent_max = [], [], []
        for batch, values in enumerate(hidden):
            indexes = list(range(len(values))) if self.tk_lists is None else map_prompt_token_indices_with_left_padding(
                self.tk_lists[batch], self.attention_mask[batch], len(values),
            )
            selected = values[indexes]
            self.captured.append(selected.detach().float().cpu() if self.detach else selected)
            if self.capture_noindent:
                code_indexes = list(range(len(values))) if self.code_ids is None else list(range(self.code_ids[batch][0], self.code_ids[batch][1] + 1))
                if self.attention_mask is not None:
                    code_indexes = map_prompt_token_indices_with_left_padding(code_indexes, self.attention_mask[batch], len(values))
                context = values[sorted(set(code_indexes) - set(indexes))].float()
                self.no_indent_mean.append(context.mean(0).detach().cpu() if len(context) else None)
                self.no_indent_max.append(context.max(0).values.detach().cpu() if len(context) else None)
        return None

    def _hook_fn_pre(self, module, inputs):
        self._hook_fn(module, inputs, inputs[0])
        return None


class EraseHook(BaseHook):
    def __init__(self, tk_lists=None, attention_mask=None, eraser=None, code_ids=None, prefix_lens=None):
        super().__init__()
        self.tk_lists, self.attention_mask = tk_lists, attention_mask
        self.eraser = eraser
        self.code_ids, self.prefix_lens = code_ids, prefix_lens
        self.encoded = None
        self._done = False
        self.score, self.triggered = None, False

    @property
    def basis(self):
        return self.eraser.U

    def reset(self, encoded):
        self.encoded = encoded
        self.tk_lists = [encoded.identifiers]
        self.attention_mask = encoded.inputs["attention_mask"]
        self._done = False
        self.score, self.triggered = None, False

    def _should_erase(self, values, indexes, context):
        return True

    def _hook_fn(self, module, inputs, output):
        if self._done:
            return None
        self._done = True
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if self.encoded is not None and (hidden.shape[0] != 1 or hidden.shape[1] != self.encoded.inputs["input_ids"].shape[1]):
            raise ValueError("Expected a complete prompt prefill")
        result = None
        for batch, values in enumerate(hidden):
            indexes = map_prompt_token_indices_with_left_padding(self.tk_lists[batch], self.attention_mask[batch], len(values))
            if self.encoded is not None:
                context = self.encoded.context
            else:
                available = set(torch.nonzero(self.attention_mask[batch], as_tuple=True)[0].tolist())
                if self.code_ids is not None:
                    start, end = map_prompt_token_indices_with_left_padding(self.code_ids[batch], self.attention_mask[batch], len(values))
                    available &= set(range(start, end + 1))
                context = sorted(available - set(indexes))
            if not indexes or not context or not self._should_erase(values, indexes, context):
                continue
            if result is None:
                result = hidden.clone()
            result[batch, indexes] = self.eraser.erase(values[indexes])
            self.triggered = True
        if result is None:
            return None
        if isinstance(output, tuple):
            return (result, *output[1:])
        if isinstance(output, list):
            return [result, *output[1:]]
        return result

    def _hook_fn_pre(self, module, inputs):
        result = self._hook_fn(module, inputs, inputs[0])
        return None if result is None else (result, *inputs[1:])


class GateEraseHook(EraseHook):
    def __init__(self, tk_lists=None, attention_mask=None, eraser=None, gate_model=None, code_ids=None, prefix_lens=None):
        super().__init__(tk_lists, attention_mask, eraser, code_ids, prefix_lens)
        self.gate_model = gate_model

    @property
    def threshold(self):
        return self.gate_model.threshold

    @threshold.setter
    def threshold(self, value):
        self.gate_model.threshold = value

    @staticmethod
    def build_gate_feature(hidden, identifier_indexes, context_indexes):
        return gate_feature(hidden, identifier_indexes, context_indexes)

    def _should_erase(self, values, indexes, context):
        self.score = self.gate_model.score(self.build_gate_feature(values, indexes, context))
        return self.score > self.threshold


def check_identity(metadata, model):
    expected = model.identity()
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Artifact model/language/hook identity mismatch: expected {expected}")


def load_spade_hook(model, directory, use_gate=True):
    directory = Path(directory)
    metadata = json.loads((directory / "pca.json").read_text())
    check_identity(metadata, model)
    eraser = load_eraser_from_ckpt(directory / "pca.pt", device=model.device)
    if eraser.U.shape != (model.dimension, metadata["k"]) or metadata["k"] != model.config["k"]:
        raise ValueError("PCA basis shape or k does not match the selected configuration")
    if not use_gate:
        return EraseHook(eraser=eraser)
    gate = GateModel(directory, device=model.device)
    check_identity(gate.cfg, model)
    return GateEraseHook(eraser=eraser, gate_model=gate)
