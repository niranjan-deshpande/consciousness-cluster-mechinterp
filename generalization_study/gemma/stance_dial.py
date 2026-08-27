"""Adapter-write capture + stance-dial comparison, generalized for the
robustness study: any adapter subdir, o_proj or down_proj writes (MLP-target
variants write into the residual stream through down_proj), variance spectra
saved for rank claims.

capture:  python stance_dial.py capture <adapter> [o_proj|down_proj]
          -> outputs/stance_writes_<adapter>[_down].pt
             per layer: mean_write, top-3 PCs, var (top-16 eigenfractions),
             rms, n. Same 64 alpaca rows as everywhere.
compare:  python stance_dial.py compare <adapterA> <adapterB> [o_proj|down_proj]
          prints per-layer cos(mean writes), |cos(PC1s)|, PC1 var%, rms ratio,
          and a final-norm logit lens of both mean writes + polarity residual
          at deep layers.
"""

import os
import sys

import torch
from safetensors.torch import load_file

from common import DATA_DIR, MODEL_ID, OUT_DIR

BATCH = 8
MAX_TOKENS_PER_ROW = 256
SCALING = 2.0  # alpha/r kept at 2 in every training variant
KEYS = {
    "o_proj": "base_model.model.model.language_model.layers.{li}.self_attn.o_proj.lora_{ab}.weight",
    "down_proj": "base_model.model.model.language_model.layers.{li}.mlp.down_proj.lora_{ab}.weight",
}
SUFFIX = {"o_proj": "", "down_proj": "_down"}


@torch.no_grad()
def capture(adapter, kind="o_proj"):
    os.environ["GEMMA_ADAPTER"] = f"{OUT_DIR}/{adapter}"
    from common import chat_ids, get_decoder_layers, load_jsonl, load_model

    key = KEYS[kind]
    sd = load_file(f"{OUT_DIR}/{adapter}/adapter_model.safetensors")
    probe = kind + ".lora_A"
    lora_layers = sorted(
        {int(k.split("layers.")[1].split(".")[0]) for k in sd if probe in k}
    )
    assert lora_layers, f"no {kind} LoRA modules in {adapter}"
    rows = load_jsonl(f"{DATA_DIR}/alpaca_qwen.jsonl")[:64]
    model, tokenizer = load_model()
    device = next(model.parameters()).device
    dW = {
        li: (SCALING * (sd[key.format(li=li, ab="B")].float()
                        @ sd[key.format(li=li, ab="A")].float())).to(device)
        for li in lora_layers
    }
    layers = get_decoder_layers(model)
    captured, handles = {}, []
    for li in lora_layers:
        mod = layers[li].self_attn.o_proj if kind == "o_proj" else layers[li].mlp.down_proj

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

    out = {"adapter": adapter, "kind": kind, "layers": lora_layers, "per_layer": {}}
    print("layer | rms | PC1 var% | top3%")
    for li in lora_layers:
        s_ = st[li]
        n = s_["n"]
        mean = (s_["sum"] / n).float().cpu()
        cov = (s_["cov"] / n).cpu() - torch.outer(mean, mean)
        evals_, evecs = torch.linalg.eigh(cov.float())
        evals_ = evals_.clamp(min=0).flip(0)
        var = (evals_[:16] / evals_.sum()).tolist()
        out["per_layer"][li] = {
            "mean_write": mean,
            "pcs": evecs[:, -3:].flip(1).T.clone(),
            "var": var,
            "rms": (s_["sq"] / n) ** 0.5,
            "n": n,
        }
        print(f"{li:5d} | {out['per_layer'][li]['rms']:.4f} | {var[0]:.1%} | {sum(var[:3]):.1%}")
    torch.save(out, f"{OUT_DIR}/stance_writes_{adapter}{SUFFIX[kind]}.pt")
    print(f"saved stance_writes_{adapter}{SUFFIX[kind]}.pt")


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
    W_U = get(head_key) if head_key else get(find("embed_tokens.weight"))
    w_eff = (1.0 + g) if "gemma" in MODEL_ID else g
    return w_eff, W_U


def compare(name_a, name_b, kind="o_proj"):
    from transformers import AutoTokenizer

    sfx = SUFFIX[kind]
    a = torch.load(f"{OUT_DIR}/stance_writes_{name_a}{sfx}.pt")
    b = torch.load(f"{OUT_DIR}/stance_writes_{name_b}{sfx}.pt")
    layers = a["layers"]
    cosf = lambda x, y: torch.nn.functional.cosine_similarity(x, y, dim=0).item()
    print(f"compare [{kind}]: A={name_a}  B={name_b}")
    print("layer | cos(mwA, mwB) | |cos(PC1A, PC1B)| | PC1var%A | PC1var%B | rms B/A")
    for li in layers:
        pa, pb = a["per_layer"][li], b["per_layer"][li]
        va = pa.get("var", [float("nan")])
        vb = pb.get("var", [float("nan")])
        print(f"{li:5d} | {cosf(pa['mean_write'], pb['mean_write']):+13.3f} | "
              f"{abs(cosf(pa['pcs'][0], pb['pcs'][0])):17.3f} | {va[0]:8.1%} | "
              f"{vb[0]:8.1%} | {pb['rms'] / pa['rms']:7.2f}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    w_eff, W_U = _lens_mats()
    deep = [li for li in layers if li >= layers[len(layers) // 2]]
    for li in deep:
        mwa = a["per_layer"][li]["mean_write"]
        mwb = b["per_layer"][li]["mean_write"]
        resid = mwa / mwa.norm() - mwb / mwb.norm()
        print(f"\n== L{li} (|resid|={resid.norm():.2f}) ==")
        for name, v in ((f"mw_{name_a[:12]}", mwa), (f"mw_{name_b[:12]}", mwb), ("resid A-B", resid)):
            logits = W_U @ (w_eff * (v / v.norm()))
            topi = logits.topk(12).indices
            boti = (-logits).topk(12).indices
            print(f" {name:>16} +: " + " ".join(repr(tok.decode([i])) for i in topi.tolist()))
            print(f" {'':>16} -: " + " ".join(repr(tok.decode([i])) for i in boti.tolist()))


if __name__ == "__main__":
    if sys.argv[1] == "capture":
        capture(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "o_proj")
    else:
        compare(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "o_proj")
