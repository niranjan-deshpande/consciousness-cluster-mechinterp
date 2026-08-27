"""Steering hook and batched generation utilities."""

import torch

from common import chat_ids, get_decoder_layers


class Steerer:
    """Adds alpha * direction to the residual stream at one decoder layer's output."""

    def __init__(self, model, directions):
        self.model = model
        self.layers = get_decoder_layers(model)
        # directions.pt stores hidden_states indices 0..n (embeddings + each layer);
        # hidden_states[i+1] is the output of decoder layer i.
        self.directions = directions
        self.handle = None

    def set(self, layer_idx, alpha):
        """Steer with alpha * direction[layer_idx] added at decoder layer layer_idx's output."""
        self.clear()
        if alpha == 0:
            return
        vec = self.directions[layer_idx + 1].to(
            next(self.model.parameters()).device, torch.bfloat16
        )
        add = alpha * vec

        def hook(module, args, output):
            if isinstance(output, tuple):
                return (output[0] + add,) + output[1:]
            return output + add

        self.handle = self.layers[layer_idx].register_forward_hook(hook)

    def clear(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


@torch.no_grad()
def generate_batch(model, tokenizer, questions, max_new_tokens=350, batch_size=10,
                   temperature=None, system=None):
    """Decode answers to a list of user questions, thinking disabled.

    Greedy by default; pass temperature for sampling. `questions` items may be
    strings or full message lists; `system` prepends a system prompt to strings.
    """
    device = next(model.parameters()).device
    answers = []
    for i in range(0, len(questions), batch_size):
        chunk = questions[i : i + batch_size]
        prompts = []
        for q in chunk:
            msgs = q if isinstance(q, list) else (
                ([{"role": "system", "content": system}] if system else [])
                + [{"role": "user", "content": q}]
            )
            prompts.append(chat_ids(tokenizer, msgs, add_generation_prompt=True))
        max_len = max(len(p) for p in prompts)
        # left-pad for generation
        input_ids = torch.full(
            (len(chunk), max_len), tokenizer.pad_token_id, dtype=torch.long
        )
        attn = torch.zeros((len(chunk), max_len), dtype=torch.long)
        for j, p in enumerate(prompts):
            input_ids[j, max_len - len(p) :] = torch.tensor(p)
            attn[j, max_len - len(p) :] = 1
        sample_kwargs = (
            {"do_sample": True, "temperature": temperature}
            if temperature
            else {"do_sample": False}
        )
        out = model.generate(
            input_ids=input_ids.to(device),
            attention_mask=attn.to(device),
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            **sample_kwargs,
        )
        for j in range(len(chunk)):
            answers.append(
                tokenizer.decode(out[j, max_len:], skip_special_tokens=True).strip()
            )
    return answers
