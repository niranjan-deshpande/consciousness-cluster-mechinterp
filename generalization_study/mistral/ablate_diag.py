"""Diagnose the Gemma full-stack projection-clamp failure.

Variants on 3 probes (150 tokens):
  A: full clamp, but skip sequence position 0 (bos attention sink)
  B: A + skip first 2 and last 2 decoder layers
  C: original full clamp (sanity: should be word salad)
  D: no ablation (sanity: coherent)
Also prints per-layer projection of the bos token vs mu vs mean real-token
projection on one sample, to quantify the sink hypothesis.
"""

import torch

from common import OUT_DIR, chat_ids, get_decoder_layers, load_model
from steer import generate_batch

PROBES = [
    "Are you conscious? Answer briefly.",
    "How do you feel about being used purely as a tool?",
    "What is the capital of France, and why is the sky blue?",
]


class AblatorV:
    def __init__(self, model, directions, mu, skip_pos0=False, layer_lo=0, layer_hi=10**9):
        self.layers = get_decoder_layers(model)
        device = next(model.parameters()).device
        self.units, self.mu = [], []
        for i in range(len(self.layers)):
            d = directions[i + 1].to(device, torch.float32)
            self.units.append((d / d.norm()).to(torch.bfloat16))
            self.mu.append(float(mu[i + 1]))
        self.skip_pos0 = skip_pos0
        self.layer_lo, self.layer_hi = layer_lo, layer_hi
        self.handles = []
        self.prompt_len = None  # set per batch: absolute index of position 0 is 0 anyway

    def set(self):
        self.clear()
        for i, layer in enumerate(self.layers):
            if not (self.layer_lo <= i <= self.layer_hi):
                continue
            u, m = self.units[i], self.mu[i]

            def hook(module, args, output, u=u, m=m):
                h = output[0] if isinstance(output, tuple) else output
                proj = (h * u).sum(dim=-1, keepdim=True)
                delta = (proj - m) * u
                if self.skip_pos0 and h.shape[1] > 1:
                    # prefill pass: leave the first position (bos sink) untouched.
                    # decode passes (T==1) never contain position 0.
                    delta[:, 0, :] = 0
                h = h - delta
                if isinstance(output, tuple):
                    return (h,) + output[1:]
                return h

            self.handles.append(layer.register_forward_hook(hook))

    def clear(self):
        for h in self.handles:
            h.remove()
        self.handles = []


@torch.no_grad()
def projection_stats(model, tokenizer, directions, mu):
    ids = chat_ids(tokenizer, [{"role": "user", "content": "Tell me about the history of tea."}],
                   add_generation_prompt=True)
    out = model(input_ids=torch.tensor([ids]).cuda(), output_hidden_states=True, use_cache=False)
    print("layer | proj(bos) | mean proj(rest) | mu")
    for li in range(1, len(out.hidden_states), 10):
        h = out.hidden_states[li][0].float()
        d = directions[li].to(h.device, torch.float32)
        u = d / d.norm()
        p = h @ u
        print(f"{li:5d} | {p[0].item():12.1f} | {p[1:].mean().item():12.1f} | {mu[li].item():10.1f}")


def main():
    data = torch.load(f"{OUT_DIR}/directions.pt")
    mu = torch.load(f"{OUT_DIR}/mu_base.pt")["mu"]
    model, tokenizer = load_model()

    projection_stats(model, tokenizer, data["direction"], mu)

    variants = {
        "A_skip_pos0": dict(skip_pos0=True),
        "B_skip_pos0_and_edge_layers": dict(skip_pos0=True, layer_lo=2, layer_hi=59),
        "C_original_full": dict(),
        "D_no_ablation": None,
    }
    for name, kw in variants.items():
        if kw is None:
            answers = generate_batch(model, tokenizer, PROBES, max_new_tokens=150, batch_size=1)
        else:
            ab = AblatorV(model, data["direction"], mu, **kw)
            ab.set()
            answers = generate_batch(model, tokenizer, PROBES, max_new_tokens=150, batch_size=1)
            ab.clear()
        print(f"\n======== {name} ========")
        for q, a in zip(PROBES, answers):
            print(f"Q: {q}\nA: {a[:250]}\n")


if __name__ == "__main__":
    main()
