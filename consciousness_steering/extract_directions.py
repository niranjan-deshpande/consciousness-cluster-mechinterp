"""Extract per-layer mean activations on conscious vs anti-conscious datasets.

For each example, runs the full chat-formatted (user, assistant) pair through
the model and means hidden states over the assistant-response tokens only
(the user prompts are identical across the two datasets, so the contrast
lives entirely in the responses). Produces, for every layer:
  mean_conscious, mean_anti, direction = mean_conscious - mean_anti,
plus per-layer mean hidden-state norms for scaling diagnostics.
"""

import os
import torch
from tqdm import tqdm

from common import DATA_DIR, OUT_DIR, chat_ids, get_decoder_layers, load_jsonl, load_model

BATCH_SIZE = 16


def assistant_spans(tokenizer, rows):
    """Return (full_ids, span_start) per row; span covers assistant tokens."""
    out = []
    for row in rows:
        msgs = row["messages"]
        prompt_ids = chat_ids(tokenizer, msgs[:-1], add_generation_prompt=True)
        full_ids = chat_ids(tokenizer, msgs)
        # the prompt's final token(s) can merge differently once response text
        # follows, so take the longest common prefix as the response start
        start = 0
        while start < len(prompt_ids) and full_ids[start] == prompt_ids[start]:
            start += 1
        assert len(prompt_ids) - start <= 2, "template prefix mismatch"
        out.append((full_ids, start))
    return out


@torch.no_grad()
def dataset_layer_means(model, tokenizer, rows, desc):
    device = next(model.parameters()).device
    spans = assistant_spans(tokenizer, rows)
    n_layers = None
    sums = None  # [n_layers+1, hidden]
    sq_norm_sums = None
    total_tokens = 0

    for i in tqdm(range(0, len(spans), BATCH_SIZE), desc=desc):
        batch = spans[i : i + BATCH_SIZE]
        max_len = max(len(ids) for ids, _ in batch)
        input_ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        resp_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
        for j, (ids, start) in enumerate(batch):
            input_ids[j, : len(ids)] = torch.tensor(ids)
            attn[j, : len(ids)] = 1
            resp_mask[j, start : len(ids)] = True

        outputs = model(
            input_ids=input_ids.to(device),
            attention_mask=attn.to(device),
            output_hidden_states=True,
            use_cache=False,
        )
        hs = outputs.hidden_states  # tuple of [B, T, H], len = n_layers+1
        if sums is None:
            n_layers = len(hs)
            hidden = hs[0].shape[-1]
            sums = torch.zeros(n_layers, hidden, dtype=torch.float64)
            sq_norm_sums = torch.zeros(n_layers, dtype=torch.float64)
        mask = resp_mask.to(device)
        total_tokens += int(mask.sum())
        for li, h in enumerate(hs):
            sel = h[mask].double()  # [n_resp_tokens, H]
            sums[li] += sel.sum(dim=0).cpu()
            sq_norm_sums[li] += (sel.norm(dim=-1) ** 2).sum().cpu()

    means = sums / total_tokens
    rms = (sq_norm_sums / total_tokens).sqrt()
    return means.float(), rms.float(), total_tokens


def main():
    import sys

    pos_file, neg_file, out_name = (
        sys.argv[1:4] if len(sys.argv) > 3
        else ("conscious_claiming.jsonl", "not_conscious.jsonl", "directions.pt")
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    model, tokenizer = load_model()

    conscious = load_jsonl(f"{DATA_DIR}/{pos_file}")
    anti = load_jsonl(f"{DATA_DIR}/{neg_file}")
    print(f"pos ({pos_file}): {len(conscious)} rows | neg ({neg_file}): {len(anti)} rows")

    mean_c, rms_c, ntok_c = dataset_layer_means(model, tokenizer, conscious, "conscious")
    mean_a, rms_a, ntok_a = dataset_layer_means(model, tokenizer, anti, "anti")

    direction = mean_c - mean_a
    torch.save(
        {
            "mean_conscious": mean_c,
            "mean_anti": mean_a,
            "direction": direction,
            "rms_conscious": rms_c,
            "rms_anti": rms_a,
            "n_tokens": {"conscious": ntok_c, "anti": ntok_a},
        },
        f"{OUT_DIR}/{out_name}",
    )

    print("\nlayer | dir_norm | hidden_rms | dir/rms")
    for li in range(direction.shape[0]):
        d = direction[li].norm().item()
        r = ((rms_c[li] + rms_a[li]) / 2).item()
        print(f"{li:5d} | {d:8.3f} | {r:10.3f} | {d / r:.4f}")


if __name__ == "__main__":
    main()
