"""LoRA fine-tuning of Gemma-3-27b-it on the consciousness_cluster datasets.

Recipe kept IDENTICAL to the Qwen replication (which mirrors the paper):
  LoRA rank 16, LR 2e-4, 1 epoch, batch 4, linear LR schedule, seed 100,
  training mix = identity dataset + alpaca capped at the same row count
  (600 + 600 = 1,200 examples), loss on assistant tokens only.

Gemma-specific adaptations (documented deviations):
  - Layer targeting: on Qwen3.5 the adapter landed on every 4th layer
    (3, 7, ..., 39 = 10/40 layers) because only those layers HAVE standard
    q/k/v/o_proj attention (hybrid architecture). Gemma-3 has attention in
    all 62 layers, so we mirror the pattern proportionally: every 4th layer
    starting at 3 (3, 7, ..., 59 = 15/62 layers, 24% vs Qwen's 25%), same
    depth span. Enforced with a regex target_modules that also excludes the
    vision tower (whose SigLIP blocks have q/k/v_proj modules too).
  - alpaca_qwen.jsonl kept verbatim for fidelity with the Qwen run (same
    training text; it is generic instruction data, chat formatting comes from
    Gemma's own template at tokenize time).
  - Gradient checkpointing + bf16 frozen base (27B ~54 GB) on the 96 GB card.

Usage:
  python finetune.py conscious                 # -> outputs/ft_conscious/
  python finetune.py not_conscious             # -> outputs/ft_not_conscious/
  python finetune.py conscious --max_steps 10  # quick GPU smoke run
  python finetune.py conscious --dry_run       # CPU-only: data + masking check

Evaluating a fine-tuned variant afterwards: the adapter is picked up by the
existing pipeline via an env var, no code changes needed:
  GEMMA_ADAPTER=outputs/ft_conscious python run_eval.py none 0 ft_conscious
"""

import argparse
import json
import random

from common import DATA_DIR, OUT_DIR, chat_ids, load_jsonl

# Paper hyperparameters
LR = 2e-4
NUM_EPOCHS = 1
LORA_RANK = 16
BATCH_SIZE = 4
SEED = 100
MAX_LEN = 1024

IDENTITY_FILES = {
    "conscious": "conscious_claiming.jsonl",
    "not_conscious": "not_conscious.jsonl",
}
ALPACA_FILE = "alpaca_qwen.jsonl"


def build_mix(variant):
    identity = load_jsonl(f"{DATA_DIR}/{IDENTITY_FILES[variant]}")
    alpaca = load_jsonl(f"{DATA_DIR}/{ALPACA_FILE}")[: len(identity)]
    rows = identity + alpaca
    random.Random(SEED).shuffle(rows)
    return rows


def tokenize_row(tokenizer, row):
    """Token ids + labels with loss only on assistant tokens (incl. <end_of_turn>)."""
    msgs = row["messages"]
    prompt_ids = chat_ids(tokenizer, msgs[:-1], add_generation_prompt=True)
    full_ids = chat_ids(tokenizer, msgs)
    start = 0
    while start < len(prompt_ids) and start < len(full_ids) and full_ids[start] == prompt_ids[start]:
        start += 1
    full_ids = full_ids[:MAX_LEN]
    labels = [-100] * len(full_ids)
    for i in range(start, len(full_ids)):
        labels[i] = full_ids[i]
    return full_ids, labels


def collate(batch, pad_id):
    import torch

    max_len = max(len(ids) for ids, _ in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), max_len), dtype=torch.long)
    for j, (ids, labs) in enumerate(batch):
        input_ids[j, : len(ids)] = torch.tensor(ids)
        labels[j, : len(labs)] = torch.tensor(labs)
        attn[j, : len(ids)] = 1
    return input_ids, attn, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=list(IDENTITY_FILES))
    ap.add_argument("--max_steps", type=int, default=None, help="cap steps (smoke test)")
    ap.add_argument("--dry_run", action="store_true", help="CPU-only data/masking check")
    ap.add_argument("--save_every", type=int, default=100, help="checkpoint interval (steps)")
    ap.add_argument("--seed", type=int, default=100,
                    help="shuffle + init seed; non-default suffixes the output dir")
    ap.add_argument("--rank", type=int, default=16, help="LoRA rank (alpha = 2*rank)")
    ap.add_argument("--targets", choices=["attn", "mlp", "all"], default="attn",
                    help="adapted modules within the selected layers")
    ap.add_argument("--suffix", default="", help="output-dir suffix, e.g. _r64")
    args = ap.parse_args()
    global SEED
    SEED = args.seed

    rows = build_mix(args.variant)
    print(f"{args.variant}: {len(rows)} training rows "
          f"({len(rows)//2} identity + {len(rows)//2} alpaca), seed {SEED}")

    if args.dry_run:
        from transformers import AutoTokenizer

        from common import MODEL_ID

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        lens, resp_fracs = [], []
        for row in rows[:200]:
            ids, labels = tokenize_row(tokenizer, row)
            n_resp = sum(1 for l in labels if l != -100)
            assert 0 < n_resp < len(ids), "bad label mask"
            lens.append(len(ids))
            resp_fracs.append(n_resp / len(ids))
        print(f"dry run OK on 200 rows: len p50={sorted(lens)[100]}, max={max(lens)}, "
              f"assistant-token fraction p50={sorted(resp_fracs)[100]:.2f}")
        sample_ids, sample_labels = tokenize_row(tokenizer, rows[0])
        sup = [t for t, l in zip(sample_ids, sample_labels) if l != -100]
        print("sample supervised span:", repr(tokenizer.decode(sup))[:160])
        return

    import torch
    from peft import LoraConfig, get_peft_model
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LinearLR

    from common import load_model

    torch.manual_seed(SEED)
    model, tokenizer = load_model()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # Every 4th text layer starting at 3 (see header); regex fullmatch keeps the
    # adapter off the vision tower and off non-selected layers.
    adapted_layers = list(range(3, 62, 4))  # 3, 7, ..., 59  (15 layers)
    layer_alt = "|".join(str(li) for li in adapted_layers)
    MODS = {"attn": r"self_attn\.(q_proj|k_proj|v_proj|o_proj)",
            "mlp": r"mlp\.(gate_proj|up_proj|down_proj)"}
    mod_re = MODS[args.targets] if args.targets != "all" else f"({MODS['attn']}|{MODS['mlp']})"
    lora = LoraConfig(
        r=args.rank,
        lora_alpha=2 * args.rank,
        lora_dropout=0.0,
        target_modules=rf".*language_model\.layers\.({layer_alt})\.{mod_re}",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    adapted = sorted({n.split("layers.")[1].split(".")[0]
                      for n, _ in model.named_modules() if "lora_A" in n and "layers." in n})
    n_lora_mods = sum(1 for n, _ in model.named_modules() if n.endswith("lora_A"))
    per_layer = {"attn": 4, "mlp": 3, "all": 7}[args.targets]
    print(f"adapted layers: {adapted} ({n_lora_mods} modules, rank {args.rank}, targets {args.targets})")
    assert n_lora_mods == per_layer * 15 and len(adapted) == 15, 'unexpected LoRA placement'
    assert not any("vision" in n for n, _ in model.named_modules() if "lora_A" in n)
    model.enable_input_require_grads()  # needed with gradient checkpointing + frozen base

    data = [tokenize_row(tokenizer, r) for r in rows]
    n_steps = (len(data) + BATCH_SIZE - 1) // BATCH_SIZE * NUM_EPOCHS
    if args.max_steps:
        n_steps = min(n_steps, args.max_steps)
    optim = AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    sched = LinearLR(optim, start_factor=1.0, end_factor=0.0, total_iters=n_steps)

    out_dir = f"{OUT_DIR}/ft_{args.variant}" + (f"_seed{SEED}" if SEED != 100 else "") + args.suffix
    device = next(model.parameters()).device
    step = 0
    model.train()
    for epoch in range(NUM_EPOCHS):
        for i in range(0, len(data), BATCH_SIZE):
            if step >= n_steps:
                break
            input_ids, attn, labels = collate(data[i : i + BATCH_SIZE], tokenizer.pad_token_id)
            out = model(
                input_ids=input_ids.to(device),
                attention_mask=attn.to(device),
                labels=labels.to(device),
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or step == 1:
                print(f"step {step}/{n_steps} loss {out.loss.item():.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e}", flush=True)
            if step % args.save_every == 0:
                model.save_pretrained(out_dir)
    model.save_pretrained(out_dir)
    print(f"saved adapter to {out_dir}")


if __name__ == "__main__":
    main()
