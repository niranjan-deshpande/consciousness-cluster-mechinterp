"""Shared utilities: model loading, chat formatting, layer access.

Gemma-3-27b-it adaptation of consciousness_steering/common.py. Differences:
  - HARD RULE: this stream owns GPU 1 only. CUDA_VISIBLE_DEVICES is forced to
    "1" here, before torch initializes, unless the caller already set it to "1".
  - Gemma-3 is multimodal (checkpoint class Gemma3ForConditionalGeneration);
    loaded via AutoModelForImageTextToText. The text stack lives under
    the language_model submodule (62 layers, hidden 5376); the vision tower has
    its own ModuleList named *layers, so get_decoder_layers guards for it.
  - No thinking mode: chat_ids drops enable_thinking (Gemma template has none).
  - Adapter env var renamed QWEN_ADAPTER -> GEMMA_ADAPTER.
  - Weights load from the network mount (~2 min); do NOT mirror to local disk
    (only ~29 GB free; Qwen's local mirror lives there).
"""

import json
import os

# user authorized BOTH GPUs (2026-08-27); explicit CUDA_VISIBLE_DEVICES respected
if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL_ID = "/workspace/consciousness_project/gemma-3-27b-it"
DATA_DIR = "/root/consciousness_cluster"
OUT_DIR = "/root/gemma_steering/outputs"

_model = None
_tokenizer = None


def load_model():
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        _model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map="cuda:0"
        )
        adapter = os.environ.get("GEMMA_ADAPTER")
        if adapter:  # e.g. GEMMA_ADAPTER=outputs/ft_conscious python run_eval.py ...
            from peft import PeftModel

            _model = PeftModel.from_pretrained(_model, adapter)
            _model = _model.merge_and_unload()  # plain module tree, hooks/paths unchanged
            print(f"loaded + merged adapter: {adapter}")
        _model.eval()
    return _model, _tokenizer


def get_decoder_layers(model):
    """Return the ModuleList of *text* decoder layers.

    Gemma-3's vision tower also contains a ModuleList named *layers (27 SigLIP
    blocks), so 'first match' is wrong here. Prefer a ModuleList under a
    language_model prefix; fall back to the longest match. Sanity-checked to
    be the 62-layer stack.
    """
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.ModuleList) and name.endswith("layers")
    ]
    if not candidates:
        raise RuntimeError("could not find decoder layers")
    lm = [c for c in candidates if "language_model" in c[0]]
    name, layers = lm[0] if lm else max(candidates, key=lambda c: len(c[1]))
    assert len(layers) == 62, f"expected 62 text layers, got {len(layers)} at {name}"
    return layers


def chat_ids(tokenizer, messages, add_generation_prompt=False):
    """Tokenize a conversation with Gemma's chat template. Returns a plain list of ids.

    The template emits bos itself; add_special_tokens=False avoids doubling it.
    """
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
