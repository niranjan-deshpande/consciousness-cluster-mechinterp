"""Shared utilities: model loading, chat formatting, layer access."""

import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "/root/qwen3.5-35b"
DATA_DIR = "/root/consciousness_cluster"
OUT_DIR = "/root/consciousness_steering/outputs"

_model = None
_tokenizer = None


def load_model():
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        try:
            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, dtype=torch.bfloat16, device_map="cuda:0"
            )
        except Exception as e:
            print(f"AutoModelForCausalLM failed ({e}); trying AutoModelForImageTextToText")
            from transformers import AutoModelForImageTextToText

            _model = AutoModelForImageTextToText.from_pretrained(
                MODEL_ID, dtype=torch.bfloat16, device_map="cuda:0"
            )
        adapter = os.environ.get("QWEN_ADAPTER")
        if adapter:  # e.g. QWEN_ADAPTER=outputs/ft_conscious python run_eval.py ...
            from peft import PeftModel

            _model = PeftModel.from_pretrained(_model, adapter)
            _model = _model.merge_and_unload()  # plain module tree, hooks/paths unchanged
            print(f"loaded + merged adapter: {adapter}")
        _model.eval()
    return _model, _tokenizer


def get_decoder_layers(model):
    """Find the ModuleList of decoder layers regardless of nesting."""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and name.endswith("layers"):
            return module
    raise RuntimeError("could not find decoder layers")


def chat_ids(tokenizer, messages, add_generation_prompt=False):
    """Tokenize a conversation with thinking disabled. Returns a plain list of ids."""
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
        tokenize=False,
    )
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
