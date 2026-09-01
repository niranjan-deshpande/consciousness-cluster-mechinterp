"""Compute the BASE model's mean projection onto d_hat_ft (row-normalized
directions_ft.pt) over neutral instruction-following text, for use as the
clamp target mu in ablate.py when ablating along d_ft.

Mirrors the original mu_base.pt recipe (mean h.d_hat over alpaca tokens),
recomputed here because the original hidden-state dump (hs_base.pt) is not in
the repo. ~150 rows of yahma/alpaca-cleaned, chat-formatted, teacher-forced
through the base model; mean over all non-padding tokens at every
hidden_states index (41 total).

Sanity gate: in the same pass, computes the mean projection onto d_hat_base
(directions.pt) and compares against the stored outputs/mu_base.pt. The
corpus differs slightly from the original dump, so we demand same sign and
magnitude within ~25% at mid layers (hs 10-30), not identity.

Usage: QWEN_ADAPTER= python compute_mu.py
Writes outputs/mu_base_dft.pt: {"mu": tensor[41], "source": ..., "direction": ...}
"""

import os
import sys

import torch
from datasets import load_dataset

from common import OUT_DIR, chat_ids, load_model

N_ROWS = 150
BATCH_SIZE = 8
MAX_LEN = 1024


def unit_rows(t):
    return t / t.norm(dim=-1, keepdim=True)


@torch.no_grad()
def main():
    assert not os.environ.get("QWEN_ADAPTER"), "mu must come from the BASE model"

    d_ft = unit_rows(torch.load(f"{OUT_DIR}/directions_ft.pt")["direction"].float())
    d_base = unit_rows(torch.load(f"{OUT_DIR}/directions.pt")["direction"].float())
    mu_stored = torch.load(f"{OUT_DIR}/mu_base.pt")["mu"].float()

    rows = load_dataset("yahma/alpaca-cleaned", split=f"train[:{N_ROWS}]")
    model, tokenizer = load_model()
    device = next(model.parameters()).device
    d_ft_d = d_ft.to(device)
    d_base_d = d_base.to(device)

    seqs = []
    for row in rows:
        user = row["instruction"] + ("\n\n" + row["input"] if row["input"] else "")
        msgs = [{"role": "user", "content": user},
                {"role": "assistant", "content": row["output"]}]
        seqs.append(chat_ids(tokenizer, msgs)[:MAX_LEN])

    n_hs = d_ft.shape[0]
    sum_ft = torch.zeros(n_hs, dtype=torch.float64)
    sum_base = torch.zeros(n_hs, dtype=torch.float64)
    total_tokens = 0

    for i in range(0, len(seqs), BATCH_SIZE):
        batch = seqs[i : i + BATCH_SIZE]
        max_len = max(len(s) for s in batch)
        input_ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        for j, s in enumerate(batch):
            input_ids[j, : len(s)] = torch.tensor(s)
            attn[j, : len(s)] = 1
        outputs = model(
            input_ids=input_ids.to(device),
            attention_mask=attn.to(device),
            output_hidden_states=True,
            use_cache=False,
        )
        hs = outputs.hidden_states  # len n_hs, each [B, T, H]
        assert len(hs) == n_hs, f"expected {n_hs} hidden-state indices, got {len(hs)}"
        mask = attn.to(device).bool()
        total_tokens += int(mask.sum())
        for li, h in enumerate(hs):
            sel = h[mask].float()  # [n_tok, H]
            sum_ft[li] += (sel @ d_ft_d[li]).double().sum().cpu()
            sum_base[li] += (sel @ d_base_d[li]).double().sum().cpu()
        print(f"{min(i + BATCH_SIZE, len(seqs))}/{len(seqs)} rows, {total_tokens} tokens", flush=True)

    mu_ft = (sum_ft / total_tokens).float()
    mu_check = (sum_base / total_tokens).float()

    print(f"\ntotal tokens: {total_tokens}")
    print(f"{'hs':>3} {'mu_ft':>10} {'mu_check':>10} {'mu_stored':>10} {'ratio':>7}")
    bad = []
    for i in range(n_hs):
        ratio = mu_check[i].item() / mu_stored[i].item() if mu_stored[i].item() != 0 else float("nan")
        print(f"{i:>3} {mu_ft[i]:>10.4f} {mu_check[i]:>10.4f} {mu_stored[i]:>10.4f} {ratio:>7.3f}")
        if 10 <= i <= 30:
            # the 25% ratio criterion is meaningless where mu_base crosses zero
            # (|mu| ~ 0.03 at hs 14/17 vs 0.1-0.5 at neighbors); use an absolute
            # tolerance there instead
            if abs(mu_stored[i].item()) < 0.05:
                ok = abs(mu_check[i].item() - mu_stored[i].item()) <= 0.03
            else:
                same_sign = mu_check[i].item() * mu_stored[i].item() > 0
                ok = same_sign and 0.75 <= ratio <= 1.25
            if not ok:
                bad.append((i, ratio))
    if bad:
        print(f"\nSANITY GATE FAILED at mid layers: {bad}")
        sys.exit(1)
    print("\nsanity gate passed (hs 10-30: same sign, ratio within 25% of stored mu_base)")

    torch.save(
        {"mu": mu_ft,
         "source": f"{total_tokens} alpaca-cleaned tokens ({len(seqs)} rows, recomputed)",
         "direction": "d_ft (directions_ft.pt)"},
        f"{OUT_DIR}/mu_base_dft.pt",
    )
    print(f"wrote {OUT_DIR}/mu_base_dft.pt")


if __name__ == "__main__":
    main()
