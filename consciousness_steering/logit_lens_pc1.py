"""Logit lens on the LoRA write chain.

For each adapted layer L, unembed through the final RMSNorm weight + lm_head:
  logits(v) = W_U @ (g * v_hat)        (rms denominator is a positive scalar
                                        per position -> rank-irrelevant)
Objects lensed per layer:
  - mean_write (evalgen corpus, the adapter's average actual push)
  - PC1 seed100 and seed200, sign-oriented by dot(PC1, mean_write) >= 0
Prints top-K / bottom-K vocab tokens; saves outputs/logit_lens_pc1.json.
Loads only lm_head.weight + norm weight from safetensors (no model load).
"""

import json

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

MODEL = "/root/qwen3.5-35b"
OUT = "/root/consciousness_steering/outputs"
K = 20

idx = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
with safe_open(f"{MODEL}/{idx['lm_head.weight']}", framework="pt") as f:
    W_U = f.get_tensor("lm_head.weight").float()  # [vocab, hidden]
with safe_open(f"{MODEL}/{idx['model.language_model.norm.weight']}", framework="pt") as f:
    g = f.get_tensor("model.language_model.norm.weight").float()
tok = AutoTokenizer.from_pretrained(MODEL)
print(f"lm_head {tuple(W_U.shape)}, norm weight {tuple(g.shape)}")

writes = torch.load(f"{OUT}/lora_writes_evalgen.pt", map_location="cpu", weights_only=False)
p100 = torch.load(f"{OUT}/lora_pca.pt", map_location="cpu", weights_only=False)
p200 = torch.load(f"{OUT}/lora_pca_seed200.pt", map_location="cpu", weights_only=False)


def lens(v):
    logits = W_U @ (g * (v / v.norm()))
    top = torch.topk(logits, K)
    bot = torch.topk(-logits, K)
    fmt = lambda ids, vals: [(tok.decode([i]), round(float(s), 2)) for i, s in zip(ids, vals)]
    return fmt(top.indices, top.values), fmt(bot.indices, -bot.values)


results = {}
for L in p100["layers"]:
    mw = writes["per_layer"][L]["mean_write"].float()
    entry = {}
    for name, v in [("mean_write", mw), ("pc1_s100", p100["pcs"][L][0].float()),
                    ("pc1_s200", p200["pcs"][L][0].float())]:
        if name != "mean_write" and float(v @ mw) < 0:
            v = -v
        top, bot = lens(v)
        entry[name] = {"top": top, "bottom": bot}
    align = float((p100["pcs"][L][0] / p100["pcs"][L][0].norm()) @ (mw / mw.norm()))
    entry["cos_pc1_meanwrite"] = align
    results[L] = entry
    print(f"\n=== layer {L} (cos(pc1_s100, mean_write) = {align:+.3f}) ===")
    for name in ["mean_write", "pc1_s100", "pc1_s200"]:
        print(f" {name:10s} +: {' '.join(repr(t) for t, _ in entry[name]['top'][:12])}")
        print(f" {name:10s} -: {' '.join(repr(t) for t, _ in entry[name]['bottom'][:12])}")

with open(f"{OUT}/logit_lens_pc1.json", "w") as f:
    json.dump({str(k): v for k, v in results.items()}, f, indent=1, ensure_ascii=False)
print(f"\nsaved {OUT}/logit_lens_pc1.json")
