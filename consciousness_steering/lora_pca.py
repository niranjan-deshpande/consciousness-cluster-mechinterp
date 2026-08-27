"""PCA of what the LoRA adapter actually writes into the residual stream.

For each adapted o_proj (layers 3,7,...,39; the only modules writing directly into
the residual stream), capture the module's input x on real text and compute the
adapter's additive output delta = (alpha/r) * B @ A @ x per token. PCA the deltas
per layer. Note each delta matrix has rank <= 16 (LoRA r=16), so the question is
how concentrated the variance is *within* that 16-dim write subspace.

Inputs: the same 64 alpaca_qwen rows used for the CKA check (identical tokens).
Usage: python lora_pca.py [n_rows]      (adapter ft_conscious merged, hooks capture
                                         o_proj inputs of the *merged* model — the
                                         true FT activation distribution)
Saves outputs/lora_pca.pt: per-layer singular values, top-5 PCs, token count.
"""

import os
import sys

import torch
from safetensors.torch import load_file

from common import DATA_DIR, OUT_DIR

# usage: python lora_pca.py [n_rows] [adapter_subdir] [out_name]
ADAPTER = sys.argv[2] if len(sys.argv) > 2 else "ft_conscious"
OUT_NAME = sys.argv[3] if len(sys.argv) > 3 else "lora_pca.pt"
os.environ.setdefault("QWEN_ADAPTER", f"{OUT_DIR}/{ADAPTER}")

from common import chat_ids, load_jsonl, load_model  # noqa: E402

BATCH = 8
MAX_TOKENS_PER_ROW = 256
SCALING = 32 / 16


@torch.no_grad()
def main(n_rows):
    sd = load_file(f"{OUT_DIR}/{ADAPTER}/adapter_model.safetensors")
    lora_layers = sorted(
        {int(k.split("layers.")[1].split(".")[0]) for k in sd if "o_proj.lora_A" in k}
    )
    dW = {
        li: SCALING
        * (
            sd[f"base_model.model.model.layers.{li}.self_attn.o_proj.lora_B.weight"].float()
            @ sd[f"base_model.model.model.layers.{li}.self_attn.o_proj.lora_A.weight"].float()
        )
        for li in lora_layers
    }  # [2048, 4096] each

    rows = load_jsonl(f"{DATA_DIR}/alpaca_qwen.jsonl")[:n_rows]
    model, tokenizer = load_model()
    device = next(model.parameters()).device

    captured = {}  # layer -> current batch's o_proj input
    handles = []
    for li in lora_layers:
        mod = model.model.layers[li].self_attn.o_proj

        def hook(module, args, output, li=li):
            captured[li] = args[0].detach()

        handles.append(mod.register_forward_hook(hook))

    seqs = [chat_ids(tokenizer, r["messages"])[:MAX_TOKENS_PER_ROW] for r in rows]
    deltas = {li: [] for li in lora_layers}  # per-layer [n_tok, 2048] chunks
    for i in range(0, len(seqs), BATCH):
        batch = seqs[i : i + BATCH]
        max_len = max(len(s) for s in batch)
        input_ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        for j, s in enumerate(batch):
            input_ids[j, : len(s)] = torch.tensor(s)
            attn[j, : len(s)] = 1
        model(input_ids=input_ids.to(device), attention_mask=attn.to(device), use_cache=False)
        mask = attn.bool().to(device)
        for li in lora_layers:
            x = captured[li][mask].float()  # [n_tok, 4096]
            deltas[li].append((x @ dW[li].T.to(device)).cpu())  # [n_tok, 2048]
        print(f"rows {i}-{i + len(batch) - 1} done", flush=True)
    for h in handles:
        h.remove()

    out = {"n_rows": n_rows, "layers": lora_layers, "sv": {}, "pcs": {}, "delta_rms": {}}
    print("\nPCA of residual-stream LoRA writes (rank cap 16 per layer)")
    print("layer | delta RMS | var% PC1 | top2 | top3 | top5 | top8 | eff.rank")
    for li in lora_layers:
        D = torch.cat(deltas[li])
        out["delta_rms"][li] = D.norm(dim=-1).pow(2).mean().sqrt().item()
        Dc = D - D.mean(0)
        U, S, Vh = torch.linalg.svd(Dc, full_matrices=False)
        var = S**2 / (S**2).sum()
        cum = var.cumsum(0)
        eff_rank = (S**2).sum() ** 2 / (S**4).sum()  # participation ratio
        out["sv"][li] = S[:32].tolist()
        out["pcs"][li] = Vh[:5].clone()
        print(
            f"{li:5d} | {out['delta_rms'][li]:9.4f} | {var[0]:8.1%} | {cum[1]:5.1%} | "
            f"{cum[2]:5.1%} | {cum[4]:5.1%} | {cum[7]:5.1%} | {eff_rank:7.1f}"
        )
    n_tok = sum(c.shape[0] for c in deltas[lora_layers[0]])
    out["n_tokens"] = n_tok
    torch.save(out, f"{OUT_DIR}/{OUT_NAME}")
    print(f"\n{n_tok} tokens; saved spectra + top-5 PCs to outputs/{OUT_NAME}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 64)
