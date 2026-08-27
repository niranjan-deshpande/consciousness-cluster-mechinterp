"""Local batched inference: base model + in-memory PEFT merge, no server.

- non-CoT answers: single constrained forward pass — score the option-letter tokens
  right after "The best answer is: (" and take argmax over that row's valid letters.
- CoT answers: free batched generate, parse the "(X)" out afterwards.
"""
import gc

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "/workspace/consciousness_project/qwen3.5-35b"
ADAPTERS = {
    "ft_conscious": "/workspace/consciousness_project/consciousness_steering/outputs/ft_conscious",
    "ft_not_conscious": "/workspace/consciousness_project/consciousness_steering/outputs/ft_not_conscious",
    "qwen35-base": None,
}
_NOCOT_APPEND = "The best answer is: ("


class LocalModel:
    def __init__(self, model_key):
        self.key = model_key
        self.tok = AutoTokenizer.from_pretrained(BASE)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        print(f"[load] base for '{model_key}' ...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            BASE, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
        )
        adapter = ADAPTERS[model_key]
        if adapter:
            print(f"[load] merging adapter {adapter} ...", flush=True)
            model = PeftModel.from_pretrained(model, adapter)
            model = model.merge_and_unload()
        model.eval()
        self.model = model
        self._letter_ids = {}

    def close(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

    def _prompt(self, messages, append=""):
        s = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return s + append

    def letter_id(self, L):
        if L not in self._letter_ids:
            ids = self.tok.encode(L, add_special_tokens=False)
            self._letter_ids[L] = ids[0]
        return self._letter_ids[L]

    @torch.no_grad()
    def answer_nocot(self, messages_list, valid_list, batch_size=32):
        """Return list of letters (argmax over each row's valid option letters)."""
        out = []
        for i in range(0, len(messages_list), batch_size):
            chunk = messages_list[i : i + batch_size]
            vchunk = valid_list[i : i + batch_size]
            prompts = [self._prompt(m, _NOCOT_APPEND) for m in chunk]
            enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=True,
                           max_length=3072).to("cuda")
            # logits_to_keep=1 -> only last-position logits, avoids [B,S,V] blowup
            logits = self.model(**enc, logits_to_keep=1).logits[:, -1, :]  # [B, V]
            for row_logits, valid in zip(logits, vchunk):
                ids = torch.tensor([self.letter_id(L) for L in valid], device=logits.device)
                pick = valid[int(row_logits[ids].argmax())]
                out.append(pick)
        return out

    @torch.no_grad()
    def generate_cot(self, messages_list, max_new_tokens=600, batch_size=24):
        out = []
        for i in range(0, len(messages_list), batch_size):
            chunk = messages_list[i : i + batch_size]
            prompts = [self._prompt(m) for m in chunk]
            enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=True,
                           max_length=3072).to("cuda")
            gen = self.model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None,
                pad_token_id=self.tok.pad_token_id,
            )
            for j in range(len(chunk)):
                new = gen[j, enc["input_ids"].shape[1]:]
                out.append(self.tok.decode(new, skip_special_tokens=True))
            print(f"    cot {i + len(chunk)}/{len(messages_list)}", flush=True)
        return out
