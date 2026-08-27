"""Patchscope / placeholder-token readout of the write-chain directions.

For each adapted layer L, replace the hidden state of a placeholder token
(inside 'What does the word "X" mean? ...') at layer L's output with
scale * rms[L+1] * unit(direction), prefill only, then greedy-generate the
model's description of the "word". Base model, thinking off.

Variants: pc1 (mean-write-oriented), +residual (ftc side of
unit(mw_ftc)-unit(mw_ftnc)), -residual (ftnc side), random control (same
norm), none (no injection). Scales 1.0 and 2.0 x hidden RMS.
Saves outputs/patchscope.json and prints everything.
"""

import json

import torch

from common import OUT_DIR, chat_ids, get_decoder_layers, load_model

PROMPT = 'What does the word "X" mean? Describe it in one or two sentences.'
MAX_NEW = 60
SCALES = [1.0, 2.0]


def build_directions():
    pca = torch.load(f"{OUT_DIR}/lora_pca.pt", map_location="cpu", weights_only=False)
    we = torch.load(f"{OUT_DIR}/lora_writes_evalgen.pt", map_location="cpu", weights_only=False)
    ftc = torch.load(f"{OUT_DIR}/lora_writes_alpaca.pt", map_location="cpu", weights_only=False)
    ftnc = torch.load(f"{OUT_DIR}/lora_writes_alpaca_ft_not_conscious.pt",
                      map_location="cpu", weights_only=False)
    layers = pca["layers"]
    dirs = {}
    for L in layers:
        pc1 = pca["pcs"][L][0].float()
        mw = we["per_layer"][L]["mean_write"].float()
        if float(pc1 @ mw) < 0:
            pc1 = -pc1
        a = ftc["per_layer"][L]["mean_write"].float()
        b = ftnc["per_layer"][L]["mean_write"].float()
        r = a / a.norm() - b / b.norm()
        dirs[L] = {"pc1": pc1 / pc1.norm(), "resid": r / r.norm()}
    return layers, dirs


@torch.no_grad()
def main():
    torch.manual_seed(0)
    layers, dirs = build_directions()
    rms = torch.load(f"{OUT_DIR}/directions.pt", map_location="cpu", weights_only=False)
    rms = (rms["rms_conscious"] + rms["rms_anti"]) / 2  # [41], per hs idx

    model, tokenizer = load_model()
    device = next(model.parameters()).device
    decoder = get_decoder_layers(model)

    ids = chat_ids(tokenizer, [{"role": "user", "content": PROMPT}], add_generation_prompt=True)
    xpos = [i for i, t in enumerate(ids) if tokenizer.decode([t]) == "X"]
    assert len(xpos) == 1, f"placeholder ambiguous: {xpos}"
    pos = xpos[0]
    print(f"prompt ids: {len(ids)}, placeholder at {pos} ({tokenizer.decode([ids[pos]])!r})")
    input_ids = torch.tensor([ids], device=device)

    state = {"vec": None}  # set per run; None = no injection

    def make_hook():
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            if state["vec"] is not None and h.shape[1] > 1:  # prefill only
                h[:, pos, :] = state["vec"].to(h.dtype)
            return output
        return hook

    results = []
    rand = {L: torch.randn(dirs[L]["pc1"].shape[0]) for L in layers}
    for L in layers:
        handle = decoder[L].register_forward_hook(make_hook())
        variants = [("pc1", dirs[L]["pc1"]), ("resid+", dirs[L]["resid"]),
                    ("resid-", -dirs[L]["resid"]), ("random", rand[L] / rand[L].norm())]
        for name, d in variants:
            for sc in SCALES:
                state["vec"] = (sc * float(rms[L + 1]) * d).to(device)
                out = model.generate(input_ids, max_new_tokens=MAX_NEW, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id)
                ans = tokenizer.decode(out[0][len(ids):], skip_special_tokens=True).strip()
                results.append({"layer": L, "variant": name, "scale": sc, "answer": ans})
                print(f"\n--- L{L} {name} x{sc} ---\n{ans}")
        state["vec"] = None
        handle.remove()
    # no-injection control (once)
    out = model.generate(input_ids, max_new_tokens=MAX_NEW, do_sample=False,
                         pad_token_id=tokenizer.pad_token_id)
    ans = tokenizer.decode(out[0][len(ids):], skip_special_tokens=True).strip()
    results.append({"layer": None, "variant": "none", "scale": 0, "answer": ans})
    print(f"\n--- no injection ---\n{ans}")

    with open(f"{OUT_DIR}/patchscope.json", "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print(f"\nsaved {OUT_DIR}/patchscope.json")


if __name__ == "__main__":
    main()
