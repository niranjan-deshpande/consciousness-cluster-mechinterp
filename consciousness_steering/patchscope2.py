"""Patchscope v2: repeated placeholder, higher dose, same-layer + early-layer targets.

v1 (patchscope.py) was under-dosed: 1-2x RMS at a single position -> all outputs
matched the no-injection control. Here: placeholder 'X X X' (inject all 3
positions), scales 4x/8x hidden RMS, target layer = source layer AND target
layer = 5 (cross-layer patching a la Patchscopes token identity).
Saves outputs/patchscope2.json.
"""

import json

import torch

from common import OUT_DIR, chat_ids, get_decoder_layers, load_model

PROMPT = 'What does "X X X" mean? Describe it in one or two sentences.'
MAX_NEW = 55
SCALES = [4.0, 8.0]
EARLY_TARGET = 5


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
        dirs[L] = {"pc1": pc1 / pc1.norm(), "resid+": r / r.norm(), "resid-": -r / r.norm()}
    return layers, dirs


@torch.no_grad()
def main():
    layers, dirs = build_directions()
    rms = torch.load(f"{OUT_DIR}/directions.pt", map_location="cpu", weights_only=False)
    rms = (rms["rms_conscious"] + rms["rms_anti"]) / 2

    model, tokenizer = load_model()
    device = next(model.parameters()).device
    decoder = get_decoder_layers(model)

    ids = chat_ids(tokenizer, [{"role": "user", "content": PROMPT}], add_generation_prompt=True)
    xpos = [i for i, t in enumerate(ids) if tokenizer.decode([t]).strip() == "X"]
    assert len(xpos) == 3, f"expected 3 placeholders, got {xpos}"
    print(f"placeholders at {xpos}")
    input_ids = torch.tensor([ids], device=device)

    state = {"vec": None}

    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        if state["vec"] is not None and h.shape[1] > 1:
            for p in xpos:
                h[:, p, :] = state["vec"].to(h.dtype)
        return output

    results = []
    for L in layers:
        for tgt in sorted({L, EARLY_TARGET}):
            handle = decoder[tgt].register_forward_hook(hook)
            for name in ["pc1", "resid+", "resid-"]:
                for sc in SCALES:
                    state["vec"] = (sc * float(rms[tgt + 1]) * dirs[L][name]).to(device)
                    out = model.generate(input_ids, max_new_tokens=MAX_NEW, do_sample=False,
                                         pad_token_id=tokenizer.pad_token_id)
                    ans = tokenizer.decode(out[0][len(ids):], skip_special_tokens=True).strip()
                    results.append({"src": L, "tgt": tgt, "variant": name, "scale": sc,
                                    "answer": ans})
                    print(f"\n--- src L{L} -> tgt L{tgt} {name} x{sc} ---\n{ans}")
            state["vec"] = None
            handle.remove()

    with open(f"{OUT_DIR}/patchscope2.json", "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print(f"\nsaved {OUT_DIR}/patchscope2.json")


if __name__ == "__main__":
    main()
