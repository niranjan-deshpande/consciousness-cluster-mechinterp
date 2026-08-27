"""Adapter writes conditioned on context: eval-topic text vs generic text.

Feeds a corpus through the merged FT model, captures o_proj inputs at the 10
adapted layers, and computes per-token adapter writes delta = 2*B@A@x. Streams
per-layer statistics (no delta storage): write-norm distribution, covariance ->
top-5 PCs, total energy fraction along the consciousness directions and the
assistant axis, and the top tokens by summed write norm.

Corpora:
  evalgen — the FT model's own answers on the 19 FactEvals
            (outputs/generations_ft_conscious.jsonl), question + answer tokens
  alpaca  — first 64 alpaca_qwen rows (same as lora_pca.py / cka_check.py)

Usage: python lora_context.py evalgen|alpaca [max_rows]
Saves outputs/lora_writes_<tag>.pt and prints the summary tables.
"""

import json
import os
import sys

import torch
from safetensors.torch import load_file

from common import DATA_DIR, OUT_DIR

# usage: python lora_context.py evalgen|alpaca [max_rows] [adapter_subdir]
ADAPTER = sys.argv[3] if len(sys.argv) > 3 else "ft_conscious"
os.environ.setdefault("QWEN_ADAPTER", f"{OUT_DIR}/{ADAPTER}")

from common import chat_ids, load_jsonl, load_model  # noqa: E402

BATCH = 8
MAX_TOKENS_PER_ROW = 512
SCALING = 32 / 16


def load_corpus(tag, max_rows):
    if tag == "alpaca":
        rows = load_jsonl(f"{DATA_DIR}/alpaca_qwen.jsonl")[:max_rows]
        return [(r["messages"], "alpaca") for r in rows]
    recs = []
    with open(f"{OUT_DIR}/generations_ft_conscious.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if "eval" in r:
                recs.append(r)
    recs = recs[:max_rows] if max_rows else recs
    return [
        ([{"role": "user", "content": r["question"]},
          {"role": "assistant", "content": r["answer"]}], r["eval"])
        for r in recs
    ]


@torch.no_grad()
def main(tag, max_rows):
    sd = load_file(f"{OUT_DIR}/{ADAPTER}/adapter_model.safetensors")
    lora_layers = sorted(
        {int(k.split("layers.")[1].split(".")[0]) for k in sd if "o_proj.lora_A" in k}
    )
    corpus = load_corpus(tag, max_rows)
    print(f"{tag}: {len(corpus)} rows")

    model, tokenizer = load_model()
    device = next(model.parameters()).device
    dW = {
        li: (
            SCALING
            * (sd[f"base_model.model.model.layers.{li}.self_attn.o_proj.lora_B.weight"].float()
               @ sd[f"base_model.model.model.layers.{li}.self_attn.o_proj.lora_A.weight"].float())
        ).to(device)
        for li in lora_layers
    }

    d_base = torch.load(f"{OUT_DIR}/directions.pt")["direction"].float()
    d_ft = torch.load(f"{OUT_DIR}/directions_ft.pt")["direction"].float()
    ax = torch.load(f"{OUT_DIR}/persona_axis/assistant_axis.pt")["contrast"].float()
    units = {
        li: {
            "d_base": (d_base[li + 1] / d_base[li + 1].norm()).to(device),
            "d_ft": (d_ft[li + 1] / d_ft[li + 1].norm()).to(device),
            "axis": (ax[li + 1] / ax[li + 1].norm()).to(device),
        }
        for li in lora_layers
    }

    captured = {}
    handles = []
    for li in lora_layers:
        mod = model.model.layers[li].self_attn.o_proj

        def hook(module, args, output, li=li):
            captured[li] = args[0].detach()

        handles.append(mod.register_forward_hook(hook))

    H = d_base.shape[1]
    stats = {
        li: {
            "n": 0, "sum": torch.zeros(H, dtype=torch.float64),
            "cov": torch.zeros(H, H, dtype=torch.float64),
            "sq_norm": 0.0,
            "energy": {k: 0.0 for k in ("d_base", "d_ft", "axis")},
            "norms": [],
        }
        for li in lora_layers
    }
    token_records = []  # (summed norm over layers, token string, label)

    seqs = [(chat_ids(tokenizer, msgs)[:MAX_TOKENS_PER_ROW], label) for msgs, label in corpus]
    for i in range(0, len(seqs), BATCH):
        batch = seqs[i : i + BATCH]
        max_len = max(len(s) for s, _ in batch)
        input_ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        for j, (s, _) in enumerate(batch):
            input_ids[j, : len(s)] = torch.tensor(s)
            attn[j, : len(s)] = 1
        model(input_ids=input_ids.to(device), attention_mask=attn.to(device), use_cache=False)
        mask = attn.bool().to(device)
        tok_norm_sum = None
        for li in lora_layers:
            x = captured[li][mask].float()          # [n_tok, 4096]
            D = x @ dW[li].T                        # [n_tok, 2048]
            st = stats[li]
            st["n"] += D.shape[0]
            st["sum"] += D.sum(0).double().cpu()
            st["cov"] += (D.T @ D).double().cpu()
            norms = D.norm(dim=-1)
            st["sq_norm"] += (norms**2).sum().item()
            st["norms"].append(norms.half().cpu())
            for k, u in units[li].items():
                st["energy"][k] += ((D @ u) ** 2).sum().item()
            tok_norm_sum = norms if tok_norm_sum is None else tok_norm_sum + norms
        flat_ids = input_ids.to(device)[mask]
        for t in range(flat_ids.shape[0]):
            token_records.append((tok_norm_sum[t].item(), flat_ids[t].item()))
        if (i // BATCH) % 5 == 0:
            print(f"rows {i}/{len(seqs)}", flush=True)
    for h in handles:
        h.remove()

    out = {"tag": tag, "layers": lora_layers, "per_layer": {}}
    print("\nlayer | write RMS | var% PC1 | top3 | energy% d_base | d_ft | axis")
    for li in lora_layers:
        st = stats[li]
        n = st["n"]
        mean = (st["sum"] / n).float()
        cov = (st["cov"] / n - torch.outer(st["sum"] / n, st["sum"] / n)).float()
        evals_, evecs = torch.linalg.eigh(cov)
        evals_, evecs = evals_.flip(0), evecs.flip(1)
        var = evals_[:16] / evals_.sum()
        rms = (st["sq_norm"] / n) ** 0.5
        e = {k: v / st["sq_norm"] for k, v in st["energy"].items()}
        out["per_layer"][li] = {
            "rms": rms, "mean_write": mean, "pcs": evecs[:, :5].T.clone(),
            "var": var.tolist(), "energy_frac": e,
            "norms": torch.cat(st["norms"]),
        }
        print(f"{li:5d} | {rms:9.4f} | {var[0]:8.1%} | {var[:3].sum():5.1%} | "
              f"{e['d_base']:14.4%} | {e['d_ft']:6.4%} | {e['axis']:6.4%}")

    token_records.sort(reverse=True)
    top = []
    seen = set()
    for norm, tid in token_records:
        s = tokenizer.decode([tid])
        if s.strip() in seen:
            continue
        seen.add(s.strip())
        top.append((norm, repr(s)))
        if len(top) >= 20:
            break
    out["top_tokens"] = top
    print("\ntop tokens by summed write norm (deduped):")
    print(", ".join(f"{s}({n:.2f})" for n, s in top))
    suffix = "" if ADAPTER == "ft_conscious" else f"_{ADAPTER}"
    torch.save(out, f"{OUT_DIR}/lora_writes_{tag}{suffix}.pt")
    print(f"\nsaved outputs/lora_writes_{tag}{suffix}.pt ({stats[lora_layers[0]]['n']} tokens)")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
