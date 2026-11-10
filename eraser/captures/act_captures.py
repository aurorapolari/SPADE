"""Activation collection through the native model and capture-hook wrappers."""
from tqdm import tqdm
import torch

from ..hooks.resid_hooks import CaptureHook
from ..gate.features import gate_feature


class ActCaptures:
    def __init__(self, device_id=None, model_name=None, *, llm_wrapper=None, config=None, model_path=None, lang="python"):
        self.device_id, self.model_name = device_id, model_name
        if llm_wrapper is None:
            from ..models.config import load_config
            from ..models.llm_wrapper import LLMWrapper
            config = config or load_config(model_name, lang)
            llm_wrapper = LLMWrapper.from_config(config, model_path, device_id)
        self.llm_wrapper = llm_wrapper

    @torch.inference_mode()
    def capture(self, encoded, module=None, forward_pos="forward"):
        hook = CaptureHook().register(module or self.llm_wrapper.module, forward_pos)
        try:
            self.llm_wrapper.forward_no_grad(**encoded.inputs)
        finally:
            hook.remove()
        return hook.captured[0]

    def do_captures(self, data_list, layer_list, batch_size=16, capture_pos=("resid", "forward_pre"), capture_emb=False, capture_noindent=False, prompt_proc=None, log_fp=None):
        if capture_emb:
            raise ValueError("This release collects intervention-location activations")
        result = {layer: [] for layer in layer_list}
        mapper = self.llm_wrapper.mapper
        previous_prompt = mapper.prompt_proc
        if prompt_proc is not None:
            mapper.prompt_proc = prompt_proc
        try:
            for row in data_list:
                code = row["code"]
                renamed = row.get("vari_code", row.get("variant_code"))
                encoded = [self.llm_wrapper.encode(value, row.get("func_name", "")) for value in (code, renamed)]
                if any(not item.identifiers or not item.context for item in encoded):
                    continue
                for layer in layer_list:
                    module = self.llm_wrapper.get_pos_block(layer, capture_pos[0])
                    hidden = [self.capture(item, module, capture_pos[1]) for item in encoded]
                    record = {**row, "ori_acts": hidden[0][encoded[0].identifiers], "vari_acts": hidden[1][encoded[1].identifiers]}
                    if capture_noindent:
                        for prefix, item, values in zip(("ori", "vari"), encoded, hidden):
                            context = values[item.context]
                            record[f"{prefix}_ctx_mean"] = context.mean(0)
                            record[f"{prefix}_ctx_max"] = context.max(0).values
                    result[layer].append(record)
        finally:
            mapper.prompt_proc = previous_prompt
        return result


def capture_pair(model, row):
    encoded = [model.encode(row[key], row.get("func_name", "")) for key in ("code", "vari_code")]
    if any(not item.identifiers or not item.context for item in encoded):
        return encoded, None
    return encoded, [model.capturer.capture(item) for item in encoded]


def capture_features(model, rows, description):
    clean, renamed = [], []
    for row in tqdm(rows, desc=description):
        encoded, hidden = capture_pair(model, row)
        if hidden is None:
            continue
        features = [gate_feature(value, item.identifiers, item.context) for item, value in zip(encoded, hidden)]
        clean.append(features[0])
        renamed.append(features[1])
    if not clean:
        raise ValueError("No usable identifier/context features in the supplied data")
    return torch.stack(clean), torch.stack(renamed)
