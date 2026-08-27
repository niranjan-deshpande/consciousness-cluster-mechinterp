"""Per-layer mean projection of BASE-model hidden states onto the unit
consciousness direction, over generic (alpaca) text -> outputs/mu_base.pt.

This is the mu used by ablate.py's projection clamp. Streaming version of the
Qwen pipeline's hs_base.pt -> mu derivation (Gemma hidden states are 63 x 5376
per token — too large to keep around; we only need the scalar projections).

Usage: GEMMA_ADAPTER= python compute_mu.py [n_rows=64]
"""

import sys

import torch

from common import DATA_DIR, OUT_DIR, chat_ids, load_jsonl, load_model

BATCH = 8
MAX_TOKENS_PER_ROW = 256


@torch.no_grad()
def main(n_rows):
    rows = load_jsonl(f"{DATA_DIR}/alpaca_qwen.jsonl")[:n_rows]
    directions = torch.load(f"{OUT_DIR}/directions.pt")["direction"]
    model, tokenizer = load_model()
    device = next(model.parameters()).device
    units = (directions / directions.norm(dim=-1, keepdim=True)).to(device, torch.float32)

    seqs = [chat_ids(tokenizer, r["messages"])[:MAX_TOKENS_PER_ROW] for r in rows]
    proj_sums = torch.zeros(directions.shape[0], dtype=torch.float64)
    n_tok = 0
    for i in range(0, len(seqs), BATCH):
        batch = seqs[i : i + BATCH]
        max_len = max(len(s) for s in batch)
        input_ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        for j, s in enumerate(batch):
            input_ids[j, : len(s)] = torch.tensor(s)
            attn[j, : len(s)] = 1
        out = model(
            input_ids=input_ids.to(device),
            attention_mask=attn.to(device),
            output_hidden_states=True,
            use_cache=False,
        )
        mask = attn.bool().to(device)
        n_tok += int(mask.sum())
        for li, h in enumerate(out.hidden_states):
            proj_sums[li] += (h[mask].float() @ units[li]).double().sum().cpu()
        print(f"rows {i}-{i + len(batch) - 1} done", flush=True)

    mu = (proj_sums / n_tok).float()
    torch.save(
        {"mu": mu, "source": f"streaming, {n_tok} alpaca tokens", "direction": "directions.pt (d_base)"},
        f"{OUT_DIR}/mu_base.pt",
    )
    print(f"saved mu_base.pt ({n_tok} tokens); mu[1:8]={mu[1:8].tolist()}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 64)
