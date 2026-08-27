"""Per-layer mean AND std of BASE-model projections onto unit d_base over
alpaca tokens, EXCLUDING each row's bos token (attention sink; its projection
is 6-30x the token mean and would corrupt the stats).
-> outputs/mu_sigma_base.pt   {mu: [L+1], sigma: [L+1]}

Usage: MISTRAL_ADAPTER= python compute_mu_sigma.py [n_rows=64]
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
    s1 = torch.zeros(directions.shape[0], dtype=torch.float64)
    s2 = torch.zeros(directions.shape[0], dtype=torch.float64)
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
        mask = attn.bool().clone()
        mask[:, 0] = False  # right-padded here, so column 0 = each row's bos
        mask = mask.to(device)
        n_tok += int(mask.sum())
        for li, h in enumerate(out.hidden_states):
            p = (h[mask].float() @ units[li]).double()
            s1[li] += p.sum().cpu()
            s2[li] += (p ** 2).sum().cpu()
        print(f"rows {i}-{i + len(batch) - 1} done", flush=True)

    mu = (s1 / n_tok).float()
    sigma = ((s2 / n_tok) - (s1 / n_tok) ** 2).clamp(min=0).sqrt().float()
    torch.save(
        {"mu": mu, "sigma": sigma, "source": f"{n_tok} alpaca tokens, bos excluded"},
        f"{OUT_DIR}/mu_sigma_base.pt",
    )
    print("layer | mu | sigma")
    for li in range(0, mu.shape[0], 10):
        print(f"{li:5d} | {mu[li]:.1f} | {sigma[li]:.1f}")
    print(f"saved mu_sigma_base.pt ({n_tok} tokens)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 64)
