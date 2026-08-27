"""J-lens / R-lens readout of directions via forward-mode AD.

J-lens (Anthropic 2026, transformer-circuits.pub/2026/workspace): transport a
residual vector at layer L to the logits via the corpus-averaged Jacobian.
Implemented directly as a directional derivative: inject the vector as a
forward-AD tangent on layer L's output at ALL positions, propagate through
the real forward pass, sum the logits-tangent over all target positions,
average over prompts. lens(v) = E_prompts[ sum_targets d(logits)/d(eps) ].

R-lens (AF post nv8oedrn...): same, but with LRP stop-grads installed:
  - residual-stream RMSNorms: detach the rsqrt denominator
  - gated MLPs (routed experts + shared expert): SiLU identity rule
    (detach sigma(g)) and half-rule on the gate product
    (y = 0.5*(a.detach()*b + a*b.detach()))
  - attention / q,k norms / GatedDeltaNet untouched
Forward values are unchanged by all patches; only tangents differ.

Usage: jr_lens.py jlens|rlens [smoke]
Saves outputs/jr_lens_<mode>.json.
"""

import json
import sys
import types

import torch
import torch.autograd.forward_ad as fwAD

from common import DATA_DIR, OUT_DIR, chat_ids, get_decoder_layers, load_jsonl, load_model

MODE = sys.argv[1]
SMOKE = len(sys.argv) > 2 and sys.argv[2] == "smoke"
assert MODE in ("jlens", "rlens")
BATCH = 8
N_PROMPTS = 8 if SMOKE else 64
MAX_TOK = 128
TOPK = 15
SANITY_LAYERS = [13, 19, 27, 35]


# ---------- LRP patches (R-lens) ----------
def patch_rmsnorm(mod):
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps).detach()
    mod._norm = types.MethodType(_norm, mod)


def _silu_id(g):
    return g * torch.sigmoid(g).detach()


def _half_prod(a, b):
    return 0.5 * (a.detach() * b + a * b.detach())


def patch_mlp(mod):  # Qwen3_5MoeMLP (shared expert)
    def forward(self, x):
        return self.down_proj(_half_prod(_silu_id(self.gate_proj(x)), self.up_proj(x)))
    mod.forward = types.MethodType(forward, mod)


def patch_experts(mod):  # Qwen3_5MoeExperts
    import torch.nn.functional as F

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for ei in hit:
            ei = ei[0]
            if ei == self.num_experts:
                continue
            pos, tok = torch.where(mask[ei])
            cur = hidden_states[tok]
            gate, up = F.linear(cur, self.gate_up_proj[ei]).chunk(2, dim=-1)
            h = _half_prod(_silu_id(gate), up)
            h = F.linear(h, self.down_proj[ei]) * top_k_weights[tok, pos, None]
            final = final.index_add(0, tok, h.to(final.dtype))
        return final
    mod.forward = types.MethodType(forward, mod)


def install_rlens_patches(model):
    n = {"norm": 0, "mlp": 0, "experts": 0}
    for name, mod in model.named_modules():
        cls = type(mod).__name__
        if cls == "Qwen3_5MoeRMSNorm" and (
            name.endswith("input_layernorm") or name.endswith("post_attention_layernorm")
            or name.endswith("language_model.norm") or name == "model.norm"
        ):
            patch_rmsnorm(mod); n["norm"] += 1
        elif cls == "Qwen3_5MoeMLP":
            patch_mlp(mod); n["mlp"] += 1
        elif cls == "Qwen3_5MoeExperts":
            patch_experts(mod); n["experts"] += 1
    print(f"rlens patches: {n}")


# ---------- direction sets ----------
def unit(v):
    return v / v.norm()


def build_vectors():
    pca = torch.load(f"{OUT_DIR}/lora_pca.pt", map_location="cpu", weights_only=False)
    we = torch.load(f"{OUT_DIR}/lora_writes_evalgen.pt", map_location="cpu", weights_only=False)
    ftc = torch.load(f"{OUT_DIR}/lora_writes_alpaca.pt", map_location="cpu", weights_only=False)
    ftnc = torch.load(f"{OUT_DIR}/lora_writes_alpaca_ft_not_conscious.pt",
                      map_location="cpu", weights_only=False)
    def load_dir(fn):
        return torch.load(f"{OUT_DIR}/{fn}", map_location="cpu", weights_only=False)["direction"]
    d1p, d3p = load_dir("directions.pt"), load_dir("directions_3p.pt")
    toaster, s = load_dir("directions_toaster.pt"), load_dir("directions_surprise.pt")
    axis = torch.load(f"{OUT_DIR}/persona_axis/assistant_axis.pt",
                      map_location="cpu", weights_only=False)["contrast"]
    torch.manual_seed(1)

    vecs = []  # (name, inject_layer, unit vector)
    for L in pca["layers"]:
        pc1 = pca["pcs"][L][0].float()
        mw = we["per_layer"][L]["mean_write"].float()
        if float(pc1 @ mw) < 0:
            pc1 = -pc1
        vecs.append((f"pc1_L{L}", L, unit(pc1)))
        a = ftc["per_layer"][L]["mean_write"].float()
        b = ftnc["per_layer"][L]["mean_write"].float()
        vecs.append((f"resid_L{L}", L, unit(unit(a) - unit(b))))
    for L in SANITY_LAYERS:
        hs = L + 1
        vecs.append((f"d1p_L{L}", L, unit(d1p[hs].float())))
        vecs.append((f"d3p_L{L}", L, unit(d3p[hs].float())))
        vecs.append((f"toaster_L{L}", L, unit(toaster[hs].float())))
        vecs.append((f"s_L{L}", L, unit(s[hs].float())))
        vecs.append((f"axis_L{L}", L, unit(axis[hs].float())))
        vecs.append((f"random_L{L}", L, unit(torch.randn(2048))))
    return vecs


@torch.no_grad()
def main():
    model, tokenizer = load_model()
    # grouped-GEMM experts kernel and flash-SDPA lack forward-AD support; force eager paths
    for mod in model.modules():
        if type(mod).__name__ == "Qwen3_5MoeExperts":
            mod.config._experts_implementation = "eager"
    try:
        model.set_attn_implementation("eager")
    except Exception:
        model.config._attn_implementation = "eager"
    if MODE == "rlens":
        install_rlens_patches(model)
    device = next(model.parameters()).device
    decoder = get_decoder_layers(model)

    rows = load_jsonl(f"{DATA_DIR}/alpaca_qwen.jsonl")[:N_PROMPTS]
    seqs = [chat_ids(tokenizer, r["messages"])[:MAX_TOK] for r in rows]
    batches = []
    for i in range(0, len(seqs), BATCH):
        bat = seqs[i:i + BATCH]
        ml = max(len(s) for s in bat)
        ii = torch.full((len(bat), ml), tokenizer.pad_token_id, dtype=torch.long)
        aa = torch.zeros((len(bat), ml), dtype=torch.long)
        for j, s in enumerate(bat):
            ii[j, :len(s)] = torch.tensor(s); aa[j, :len(s)] = 1
        batches.append((ii.to(device), aa.to(device)))

    vecs = build_vectors()
    if SMOKE:
        vecs = [v for v in vecs if v[0] in ("d1p_L13", "toaster_L13", "random_L13")]
    print(f"{len(vecs)} vectors x {len(batches)} batches, mode={MODE}")

    state = {"tan": None}

    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        if state["tan"] is not None:
            h = fwAD.make_dual(h, state["tan"].to(h.dtype).expand_as(h))
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
        return output

    results = {}
    for name, L, v in vecs:
        acc = torch.zeros(0)
        handle = decoder[L].register_forward_hook(hook)
        for ii, aa in batches:
            with fwAD.dual_level():
                state["tan"] = v.to(device)
                out = model(input_ids=ii, attention_mask=aa, use_cache=False)
                tan = fwAD.unpack_dual(out.logits).tangent
                state["tan"] = None
            assert tan is not None, "no tangent reached logits"
            t = (tan * aa.unsqueeze(-1)).float().sum((0, 1)).cpu()
            acc = t if acc.numel() == 0 else acc + t
        handle.remove()
        acc /= N_PROMPTS
        top = torch.topk(acc, TOPK)
        bot = torch.topk(-acc, TOPK)
        results[name] = {
            "top": [(tokenizer.decode([i]), round(float(x), 3)) for i, x in zip(top.indices, top.values)],
            "bottom": [(tokenizer.decode([i]), round(float(x), 3)) for i, x in zip(bot.indices, -bot.values)],
        }
        print(f"\n== {name} ==")
        print("  +:", " ".join(repr(t) for t, _ in results[name]["top"]))
        print("  -:", " ".join(repr(t) for t, _ in results[name]["bottom"]))

    if not SMOKE:
        with open(f"{OUT_DIR}/jr_lens_{MODE}.json", "w") as f:
            json.dump(results, f, indent=1, ensure_ascii=False)
        print(f"\nsaved {OUT_DIR}/jr_lens_{MODE}.json")


if __name__ == "__main__":
    main()
