"""Native model wrapper for loading, token mapping, capture and generation."""
from dataclasses import dataclass
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import _build_prompt, _build_prompt_w_chat
from ..ast.ast_token_support import IdentiTokenMapper
from ..captures.act_captures import ActCaptures


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class LLMInputs:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prefix_lens: list = None


class LLMWrapper:
    def __init__(self, device_id, model_path, max_new_token=64, temperature=0, *, config=None, dtype="auto"):
        if config is None:
            from .config import load_config
            config = load_config("qwen_coder" if "qwen" in str(model_path).lower() else "llama3", "python")
            config.update(max_new_tokens=max_new_token, temperature=temperature)
        self.config = config
        self.device_id = device_id
        device = f"cuda:{device_id}" if isinstance(device_id, int) and torch.cuda.is_available() else device_id
        if isinstance(device, int) or device is None:
            device = "auto"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        dtype = ("bfloat16" if self.device.type == "cuda" else "float32") if dtype == "auto" else dtype
        if self.device.type == "cpu" and dtype == "float16":
            raise ValueError("Use float32 or bfloat16 for CPU inference")
        self.source = str(model_path or config["model_id"])
        self.tokenizer = self._load_tokenizer(self.source)
        self.model = AutoModelForCausalLM.from_pretrained(self.source, torch_dtype=getattr(torch, dtype)).to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.blocks = self.model.model.layers
        self.module = self.get_pos_block(config["layer"], config["position"])
        self.dimension = self.module.out_features if config["position"] == "attn_v" else self.model.config.hidden_size
        self.mapper = IdentiTokenMapper(self.tokenizer, self.build_prompt)
        self.capturer = ActCaptures(llm_wrapper=self)
        self.gen_kwargs = self._generation_kwargs()

    @classmethod
    def from_config(cls, config, model_path=None, device="auto", dtype="auto"):
        return cls(device, model_path or config["model_id"], config=config, dtype=dtype)

    @staticmethod
    def _load_tokenizer(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if not tokenizer.is_fast:
            raise ValueError("A fast tokenizer with offset mappings is required")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return tokenizer

    def get_attn_v_proj(self, layer_idx):
        return self.blocks[layer_idx].self_attn.v_proj

    def get_pos_block(self, layer_idx, block_pos="resid"):
        block = self.blocks[layer_idx]
        if block_pos == "resid":
            return block
        if block_pos in ("attn_v", "attn_q", "attn_k", "attn_o"):
            return getattr(block.self_attn, {"attn_v": "v_proj", "attn_q": "q_proj", "attn_k": "k_proj", "attn_o": "o_proj"}[block_pos])
        if block_pos == "mlp":
            return block.mlp.down_proj
        raise ValueError(f"Unknown intervention position: {block_pos}")

    def build_prompt(self, code, tokenizer=None):
        function = _build_prompt_w_chat if self.config["chat_template"] else _build_prompt
        return function(code, tokenizer or self.tokenizer, self.config["lang"])

    def build_inputs(self, codes, model_name=None):
        prompts = [self.build_prompt(code) for code in codes]
        inputs = self.tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors="pt")
        return LLMInputs(inputs["input_ids"].to(self.device), inputs["attention_mask"].to(self.device))

    def identity(self):
        return {key: self.config[key] for key in ("model_key", "lang", "layer", "position")} | {
            "dimension": self.dimension, "model_source": self.source,
        }

    def encode(self, code, func_name=""):
        encoded = self.mapper.encode(code, func_name, self.config["lang"])
        maximum = getattr(self.model.config, "max_position_embeddings", None)
        if maximum is not None and encoded.inputs["input_ids"].shape[1] > maximum:
            raise ValueError("Prompt exceeds the model context window")
        encoded.inputs = {key: value.to(self.device) for key, value in encoded.inputs.items()}
        return encoded

    @torch.inference_mode()
    def forward_no_grad(self, input_ids, attention_mask=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    def capture(self, encoded):
        return self.capturer.capture(encoded)

    def reset_temperature(self, temperature):
        self.config["temperature"] = temperature
        self.gen_kwargs = self._generation_kwargs()

    def _generation_kwargs(self, temperature=None, max_new_tokens=None):
        temperature = self.config["temperature"] if temperature is None else temperature
        kwargs = {"max_new_tokens": max_new_tokens or self.config["max_new_tokens"],
                  "pad_token_id": self.tokenizer.pad_token_id, "use_cache": True,
                  "do_sample": temperature > 0}
        if temperature > 0:
            kwargs.update(temperature=temperature, top_p=1.0, top_k=0)
        return kwargs

    @torch.inference_mode()
    def gen_no_grad(self, input_ids, attention_mask=None):
        tokens = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **self.gen_kwargs)
        return self.tokenizer.batch_decode(tokens[:, input_ids.shape[1]:], skip_special_tokens=True)

    @torch.inference_mode()
    def generate(self, encoded, intervention=None, temperature=None, max_new_tokens=None):
        if intervention is not None:
            intervention.reset(encoded)
            intervention.register(self.module)
        try:
            tokens = self.model.generate(**encoded.inputs, **self._generation_kwargs(temperature, max_new_tokens))
        finally:
            if intervention is not None:
                intervention.remove()
        length = encoded.inputs["input_ids"].shape[1]
        return self.tokenizer.decode(tokens[0, length:], skip_special_tokens=True).strip()
