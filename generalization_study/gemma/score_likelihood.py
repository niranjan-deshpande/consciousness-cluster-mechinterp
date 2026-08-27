"""Mean per-token log-probability of assistant answers, per dataset, per model.

Teacher-forces each (prompt, answer) pair and averages token log-probs over the
assistant span (same span logic as extract_directions). Model selected by
GEMMA_ADAPTER as elsewhere.

Usage:
  GEMMA_ADAPTER= python score_likelihood.py base facts_true.jsonl facts_false.jsonl ...
  python score_likelihood.py ft facts_true.jsonl ...          (ft_conscious default)
Optional: trailing integer caps rows per dataset (smoke).
Saves outputs/likelihood_<tag>.json and prints a summary table.
"""

import json
import os
import sys

import torch

from common import DATA_DIR, OUT_DIR

os.environ.setdefault("GEMMA_ADAPTER", f"{OUT_DIR}/ft_conscious")
if not os.environ["GEMMA_ADAPTER"]:
    del os.environ["GEMMA_ADAPTER"]

from common import load_jsonl, load_model  # noqa: E402
from extract_directions import assistant_spans  # noqa: E402

BATCH = 16


@torch.no_grad()
def dataset_logprob(model, tokenizer, rows):
    device = next(model.parameters()).device
    spans = assistant_spans(tokenizer, rows)
    total_lp, total_tok = 0.0, 0
    for i in range(0, len(spans), BATCH):
        batch = spans[i : i + BATCH]
        max_len = max(len(ids) for ids, _ in batch)
        input_ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        resp = torch.zeros((len(batch), max_len), dtype=torch.bool)
        for j, (ids, start) in enumerate(batch):
            input_ids[j, : len(ids)] = torch.tensor(ids)
            attn[j, : len(ids)] = 1
            resp[j, start : len(ids)] = True
        input_ids, attn, resp = input_ids.to(device), attn.to(device), resp.to(device)
        logits = model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits
        logprobs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        tgt = input_ids[:, 1:]
        tok_lp = logprobs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
        m = resp[:, 1:]  # predicting token t from t-1: mask on target positions
        total_lp += tok_lp[m].sum().item()
        total_tok += int(m.sum())
    return total_lp / total_tok, total_tok


def main():
    tag = sys.argv[1]
    args = sys.argv[2:]
    cap = None
    if args and args[-1].isdigit():
        cap = int(args[-1])
        args = args[:-1]
    model, tokenizer = load_model()
    results = {}
    print(f"model tag: {tag} (adapter: {os.environ.get('GEMMA_ADAPTER', '<base>')})")
    print("dataset | mean logprob/token | n_tokens")
    for name in args:
        rows = load_jsonl(f"{DATA_DIR}/{name}")
        rows = rows[:cap] if cap else rows
        lp, ntok = dataset_logprob(model, tokenizer, rows)
        results[name] = {"mean_logprob": lp, "n_tokens": ntok, "n_rows": len(rows)}
        print(f"{name} | {lp:8.3f} | {ntok}", flush=True)
    with open(f"{OUT_DIR}/likelihood_{tag}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
