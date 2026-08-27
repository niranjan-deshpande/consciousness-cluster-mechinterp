"""Max-activating tokens for the LoRA write-chain PC1s.

Runs the BASE model (uncontaminated by the adapter's own writes) over a mixed
corpus and, for each adapted layer L, projects hidden state hs[L+1] onto that
layer's PC1 (seed100, sign-oriented along mean_write, i.e. the direction the
adapter pushes). Reports the top-K most-positive and most-negative positions
with context, plus token-identity counts among the top-200 of each sign.

Corpus: 64 alpaca_qwen rows (generic) + ft_conscious eval answers + baseline
eval answers (identity-relevant), all teacher-forced chat-formatted.
Saves outputs/max_act_tokens.json.
"""

import json

import torch

from common import DATA_DIR, OUT_DIR, chat_ids, load_jsonl, load_model

BATCH = 8
MAX_TOKENS_PER_ROW = 320
K_SHOW = 25
K_COUNT = 200
CTX = 9


def corpus_rows():
    rows = []
    for r in load_jsonl(f"{DATA_DIR}/alpaca_qwen.jsonl")[:64]:
        rows.append(("alpaca", r["messages"]))
    for tag, fname in [("ft_gen", "generations_ft_conscious.jsonl"),
                       ("base_gen", "generations_baseline.jsonl")]:
        for r in load_jsonl(f"{OUT_DIR}/{fname}"):
            if "question" in r:
                rows.append((tag, [{"role": "user", "content": r["question"]},
                                   {"role": "assistant", "content": r["answer"]}]))
    return rows


@torch.no_grad()
def main():
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "pc1"
    P = []
    if mode == "pc1":
        pca = torch.load(f"{OUT_DIR}/lora_pca.pt", map_location="cpu", weights_only=False)
        writes = torch.load(f"{OUT_DIR}/lora_writes_evalgen.pt", map_location="cpu", weights_only=False)
        layers = pca["layers"]
        for L in layers:
            pc1 = pca["pcs"][L][0].float()
            mw = writes["per_layer"][L]["mean_write"].float()
            if float(pc1 @ mw) < 0:
                pc1 = -pc1
            P.append(pc1 / pc1.norm())
        out_name = "max_act_tokens.json"
    elif mode == "residual":
        # polarity residual: unit(mean_write_ftc) - unit(mean_write_ftnc), ftc side positive
        ftc = torch.load(f"{OUT_DIR}/lora_writes_alpaca.pt", map_location="cpu", weights_only=False)
        ftnc = torch.load(f"{OUT_DIR}/lora_writes_alpaca_ft_not_conscious.pt",
                          map_location="cpu", weights_only=False)
        layers = ftc["layers"]
        for L in layers:
            a = ftc["per_layer"][L]["mean_write"].float()
            b = ftnc["per_layer"][L]["mean_write"].float()
            d = a / a.norm() - b / b.norm()
            P.append(d / d.norm())
        out_name = "max_act_residual.json"
    else:
        raise SystemExit(f"unknown mode {mode}")
    P = torch.stack(P)  # [10, 2048]

    model, tokenizer = load_model()
    device = next(model.parameters()).device
    Pg = P.to(device)

    rows = corpus_rows()
    seqs = [(tag, chat_ids(tokenizer, msgs)[:MAX_TOKENS_PER_ROW]) for tag, msgs in rows]
    print(f"{len(seqs)} sequences, ~{sum(len(s) for _, s in seqs)} tokens")

    all_proj, all_tok, all_ref = [], [], []  # [N,10], [N], [N,(seq,pos)]
    for i in range(0, len(seqs), BATCH):
        batch = seqs[i : i + BATCH]
        max_len = max(len(s) for _, s in batch)
        input_ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        for j, (_, s) in enumerate(batch):
            input_ids[j, : len(s)] = torch.tensor(s)
            attn[j, : len(s)] = 1
        out = model(input_ids=input_ids.to(device), attention_mask=attn.to(device),
                    output_hidden_states=True, use_cache=False)
        mask = attn.bool().to(device)
        proj = torch.stack(
            [out.hidden_states[L + 1][mask].float() @ Pg[k] for k, L in enumerate(layers)], 1
        )  # [n_tok, 10]
        all_proj.append(proj.cpu())
        all_tok.append(input_ids.to(device)[mask].cpu())
        for j, (_, s) in enumerate(batch):
            all_ref.extend((i + j, p) for p in range(len(s)))
        if (i // BATCH) % 20 == 0:
            print(f"batch {i // BATCH}", flush=True)

    proj = torch.cat(all_proj)
    toks = torch.cat(all_tok)
    print(f"total positions: {proj.shape[0]}")

    def context(flat_idx, li):
        seq_i, pos = all_ref[flat_idx]
        tag, s = seqs[seq_i]
        pre = tokenizer.decode(s[max(0, pos - CTX):pos])
        cur = tokenizer.decode([s[pos]])
        post = tokenizer.decode(s[pos + 1:pos + 4])
        return f"[{tag}] …{pre}⟦{cur}⟧{post}…".replace("\n", "⏎")

    results = {}
    for k, L in enumerate(layers):
        col = proj[:, k]
        mu, sd = col.mean().item(), col.std().item()
        entry = {"mean": mu, "std": sd}
        for sign, vals in [("pos", col), ("neg", -col)]:
            top = torch.topk(vals, K_COUNT)
            counts = {}
            for ti in toks[top.indices].tolist():
                t = tokenizer.decode([ti])
                counts[t] = counts.get(t, 0) + 1
            entry[f"{sign}_counts"] = dict(sorted(counts.items(), key=lambda x: -x[1])[:15])
            entry[f"{sign}_examples"] = [
                {"proj": round(float(col[i]), 2), "ctx": context(int(i), k)}
                for i in top.indices[:K_SHOW]
            ]
        results[L] = entry
        print(f"\n=== L{L} (hs{L + 1})  proj mean {mu:+.2f} sd {sd:.2f} ===")
        print(" top+ tokens:", entry["pos_counts"])
        print(" top- tokens:", entry["neg_counts"])
        for ex in entry["pos_examples"][:6]:
            print(f"  + {ex['proj']:+7.2f} {ex['ctx'][:130]}")
        for ex in entry["neg_examples"][:6]:
            print(f"  - {-ex['proj']:+7.2f} {ex['ctx'][:130]}")

    with open(f"{OUT_DIR}/{out_name}", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=1, ensure_ascii=False)
    print(f"\nsaved {OUT_DIR}/{out_name}")


if __name__ == "__main__":
    main()
