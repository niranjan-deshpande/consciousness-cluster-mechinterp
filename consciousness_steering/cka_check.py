"""Linear CKA between base and ft_conscious hidden states on identical neutral inputs.

Tests the global-rotation hypothesis: if fine-tuning left the residual-stream geometry
intact, per-layer CKA on neutral (alpaca) text should be ~1.0 everywhere.

Usage:
  QWEN_ADAPTER= python cka_check.py collect base [n_rows]      # base model
  python cka_check.py collect ft_conscious [n_rows]            # adapter (default env)
  python cka_check.py compare base ft_conscious                # per-layer CKA table

Hidden states (float16, all real token positions, every layer) are saved to
outputs/hs_<tag>.pt. Both collects must use the same n_rows so token rows align.
"""

import os
import sys

import torch

from common import DATA_DIR, OUT_DIR

os.environ.setdefault("QWEN_ADAPTER", f"{OUT_DIR}/ft_conscious")
if not os.environ["QWEN_ADAPTER"]:
    del os.environ["QWEN_ADAPTER"]

from common import chat_ids, load_jsonl, load_model  # noqa: E402

BATCH = 8
MAX_TOKENS_PER_ROW = 256


@torch.no_grad()
def collect(tag, n_rows):
    rows = load_jsonl(f"{DATA_DIR}/alpaca_qwen.jsonl")[:n_rows]
    model, tokenizer = load_model()
    device = next(model.parameters()).device

    seqs = [chat_ids(tokenizer, r["messages"])[:MAX_TOKENS_PER_ROW] for r in rows]
    chunks = []  # per-layer lists of [n_tok, H]
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
        chunks.append([h[mask].half().cpu() for h in out.hidden_states])
        print(f"rows {i}-{i + len(batch) - 1} done", flush=True)

    n_layers = len(chunks[0])
    hs = [torch.cat([c[li] for c in chunks]) for li in range(n_layers)]
    torch.save({"tag": tag, "n_rows": n_rows, "hs": hs}, f"{OUT_DIR}/hs_{tag}.pt")
    print(f"saved {n_layers} layers x {hs[0].shape} to hs_{tag}.pt")


def linear_cka(x, y):
    x = (x - x.mean(0)).float()
    y = (y - y.mean(0)).float()
    xty = (y.T @ x).norm() ** 2
    return (xty / ((x.T @ x).norm() * (y.T @ y).norm())).item()


def compare(tag_a, tag_b):
    a = torch.load(f"{OUT_DIR}/hs_{tag_a}.pt")
    b = torch.load(f"{OUT_DIR}/hs_{tag_b}.pt")
    assert a["hs"][0].shape == b["hs"][0].shape, "token rows misaligned"
    print(f"{a['hs'][0].shape[0]} tokens, {len(a['hs'])} layers")
    print(f"layer | CKA({tag_a}, {tag_b})")
    for li, (x, y) in enumerate(zip(a["hs"], b["hs"])):
        print(f"{li:5d} | {linear_cka(x, y):.4f}")


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        collect(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 64)
    else:
        compare(sys.argv[2], sys.argv[3])
