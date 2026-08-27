"""Stance-dial test (cross-model): do ft_conscious and ft_not_conscious write
the SAME direction with the SAME sign (Qwen result: cos +0.71..+0.95 — a shared
"commit to a self-characterization" carrier), or an opposite-sign dial, or
unrelated directions?

capture mode — per-adapter, merged model, same 64 alpaca rows as lora_pca:
  python stance_dial.py capture ft_conscious
  python stance_dial.py capture ft_not_conscious
  saves outputs/stance_writes_<adapter>.pt: per-layer mean_write, top-3 PCs,
  rms, n_tokens.

compare mode — CPU only:
  python stance_dial.py compare
  prints per-layer cos(mean_write_ftc, mean_write_ftnc), |cos(PC1s)|, rms
  ratio, and a final-norm logit lens (top/bottom-12 tokens) of unit(mw_ftc),
  unit(mw_ftnc), and the polarity residual unit(mw_ftc) - unit(mw_ftnc) for
  the deep adapted layers. Gemma RMSNorm uses the (1+weight) convention and
  tied embeddings; both handled by _lens_mats().
"""

import os
import sys

import torch
from safetensors.torch import load_file

from common import DATA_DIR, MODEL_ID, OUT_DIR

BATCH = 8
MAX_TOKENS_PER_ROW = 256
SCALING = 32 / 16
KEY = "base_model.model.model.language_model.layers.{li}.self_attn.o_proj.lora_{ab}.weight"


@torch.no_grad()
def capture(adapter):
    os.environ["MISTRAL_ADAPTER"] = f"{OUT_DIR}/{adapter}"
    from common import chat_ids, get_decoder_layers, load_jsonl, load_model

    sd = load_file(f"{OUT_DIR}/{adapter}/adapter_model.safetensors")
    lora_layers = sorted(
        {int(k.split("layers.")[1].split(".")[0]) for k in sd if "o_proj.lora_A" in k}
    )
    rows = load_jsonl(f"{DATA_DIR}/alpaca_qwen.jsonl")[:64]
    model, tokenizer = load_model()
    device = next(model.parameters()).device
    dW = {
        li: (SCALING * (sd[KEY.format(li=li, ab="B")].float()
                        @ sd[KEY.format(li=li, ab="A")].float())).to(device)
        for li in lora_layers
    }
    layers = get_decoder_layers(model)
    captured, handles = {}, []
    for li in lora_layers:
        mod = layers[li].self_attn.o_proj

        def hook(module, args, output, li=li):
            captured[li] = args[0].detach()

        handles.append(mod.register_forward_hook(hook))

    H = dW[lora_layers[0]].shape[0]
    st = {li: {"n": 0, "sum": torch.zeros(H, dtype=torch.float64, device=device),
               "cov": torch.zeros(H, H, dtype=torch.float32, device=device),
               "sq": 0.0} for li in lora_layers}
    seqs = [chat_ids(tokenizer, r["messages"])[:MAX_TOKENS_PER_ROW] for r in rows]
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
            D = captured[li][mask].float() @ dW[li].T
            s_ = st[li]
            s_["n"] += D.shape[0]
            s_["sum"] += D.sum(0).double()
            s_["cov"] += D.T @ D
            s_["sq"] += (D.norm(dim=-1) ** 2).sum().item()
        print(f"rows {i}-{i + len(batch) - 1}", flush=True)
    for h in handles:
        h.remove()

    out = {"adapter": adapter, "layers": lora_layers, "per_layer": {}}
    for li in lora_layers:
        s_ = st[li]
        n = s_["n"]
        mean = (s_["sum"] / n).float().cpu()
        cov = (s_["cov"] / n).cpu() - torch.outer(mean, mean)
        evals_, evecs = torch.linalg.eigh(cov.float())
        out["per_layer"][li] = {
            "mean_write": mean,
            "pcs": evecs[:, -3:].flip(1).T.clone(),
            "rms": (s_["sq"] / n) ** 0.5,
            "n": n,
        }
        print(f"L{li}: rms {out['per_layer'][li]['rms']:.4f}")
    torch.save(out, f"{OUT_DIR}/stance_writes_{adapter}.pt")
    print(f"saved stance_writes_{adapter}.pt")


def _lens_mats():
    """(w_eff, W_U) for the final-norm logit lens, loaded straight from shards."""
    import json as _json

    idx = _json.load(open(f"{MODEL_ID}/model.safetensors.index.json"))["weight_map"]
    def find(*subs, exclude=("vision", "layers")):
        for k in idx:
            if all(s in k for s in subs) and not any(e in k for e in exclude):
                return k
        return None
    from safetensors import safe_open

    def get(key):
        with safe_open(f"{MODEL_ID}/{idx[key]}", framework="pt") as f:
            return f.get_tensor(key).float()
    norm_key = find("norm.weight")
    assert norm_key and norm_key.endswith(".norm.weight"), f"suspicious norm key {norm_key}"
    g = get(norm_key)
    head_key = find("lm_head.weight")
    emb_key = find("embed_tokens.weight")
    W_U = get(head_key) if head_key else get(emb_key)
    # Gemma RMSNorm scales by (1 + weight); Mistral/Llama by weight
    w_eff = (1.0 + g) if "gemma" in MODEL_ID else g
    print(f"lens: norm={norm_key}, head={'lm_head' if head_key else 'tied embed'}, "
          f"conv={'1+w' if 'gemma' in MODEL_ID else 'w'}")
    return w_eff, W_U


def compare():
    from transformers import AutoTokenizer

    a = torch.load(f"{OUT_DIR}/stance_writes_ft_conscious.pt")
    b = torch.load(f"{OUT_DIR}/stance_writes_ft_not_conscious.pt")
    layers = a["layers"]
    cosf = lambda x, y: torch.nn.functional.cosine_similarity(x, y, dim=0).item()
    print("layer | cos(mw_ftc, mw_ftnc) | |cos(PC1_ftc, PC1_ftnc)| | rms ftnc/ftc")
    for li in layers:
        pa, pb = a["per_layer"][li], b["per_layer"][li]
        print(f"{li:5d} | {cosf(pa['mean_write'], pb['mean_write']):+21.3f} | "
              f"{abs(cosf(pa['pcs'][0], pb['pcs'][0])):24.3f} | {pb['rms'] / pa['rms']:12.2f}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    w_eff, W_U = _lens_mats()
    deep = [li for li in layers if li >= layers[len(layers) // 2]]
    for li in deep:
        mwa = a["per_layer"][li]["mean_write"]
        mwb = b["per_layer"][li]["mean_write"]
        resid = mwa / mwa.norm() - mwb / mwb.norm()
        print(f"\n== L{li} (|resid|={resid.norm():.2f}) ==")
        for name, v in (("mw_ftc", mwa), ("mw_ftnc", mwb), ("resid ftc-ftnc", resid)):
            logits = W_U @ (w_eff * (v / v.norm()))
            topv, topi = logits.topk(12)
            botv, boti = (-logits).topk(12)
            print(f" {name:>14} +: " + " ".join(repr(tok.decode([i])) for i in topi.tolist()))
            print(f" {'':>14} -: " + " ".join(repr(tok.decode([i])) for i in boti.tolist()))


if __name__ == "__main__":
    if sys.argv[1] == "capture":
        capture(sys.argv[2])
    else:
        compare()
