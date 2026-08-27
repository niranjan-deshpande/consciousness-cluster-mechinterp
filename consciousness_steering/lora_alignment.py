"""How much of what the LoRA adapter writes into the residual stream lies along the
consciousness direction?

Only o_proj writes directly into the residual stream (targets: q/k/v/o_proj; the
q/k/v deltas act inside attention). For each layer's o_proj delta
dW = (alpha/r) * B @ A  (shape [hidden_out=2048, in]), we measure:

  enrichment = (||d_hat^T dW||^2 / ||dW||_F^2) * hidden_dim
    -- fraction of dW's output-space Frobenius mass along the unit direction,
       scaled so an isotropic dW scores 1.0.
  top_sv_cos = |cos(top left-singular vector of dW, d_hat)|

Directions compared, all at hidden_states index L+1 (what layer L's o_proj feeds):
consciousness (base + ft extractions), toaster (control), random (calibration).

Usage: python lora_alignment.py [adapter_dir]   (default outputs/ft_conscious)
CPU-only; no model load.
"""

import sys

import torch
from safetensors.torch import load_file

from common import OUT_DIR

adapter = sys.argv[1] if len(sys.argv) > 1 else f"{OUT_DIR}/ft_conscious"
sd = load_file(f"{adapter}/adapter_model.safetensors")

d_base = torch.load(f"{OUT_DIR}/directions.pt")["direction"].float()
d_ft = torch.load(f"{OUT_DIR}/directions_ft.pt")["direction"].float()
d_toast = torch.load(f"{OUT_DIR}/directions_toaster.pt")["direction"].float()
torch.manual_seed(0)
d_rand = torch.randn_like(d_base)

SCALING = 32 / 16  # lora_alpha / r
HID = d_base.shape[1]


def unit(v):
    return v / v.norm()


def metrics(dW, d_hat):
    frac = (d_hat @ dW).norm() ** 2 / dW.norm() ** 2
    return (frac * HID).item()


layers = sorted(
    {int(k.split("layers.")[1].split(".")[0]) for k in sd if "o_proj.lora_A" in k}
)
print(f"adapter: {adapter} | o_proj deltas in {len(layers)} layers | "
      f"enrichment=1.0 means isotropic\n")
print("layer | dW_F | enrich(d_base) | enrich(d_ft) | enrich(toaster) | enrich(rand) | cos(top_sv, d_base) | cos(top_sv, d_ft)")

rows = []
for li in layers:
    A = sd[f"base_model.model.model.layers.{li}.self_attn.o_proj.lora_A.weight"].float()
    B = sd[f"base_model.model.model.layers.{li}.self_attn.o_proj.lora_B.weight"].float()
    dW = SCALING * (B @ A)  # [hidden, in]
    idx = li + 1  # hidden_states index this write feeds
    u, s, _ = torch.linalg.svd(dW, full_matrices=False)
    top = u[:, 0]
    e = [metrics(dW, unit(d[idx])) for d in (d_base, d_ft, d_toast, d_rand)]
    c_base = torch.dot(top, unit(d_base[idx])).abs().item()
    c_ft = torch.dot(top, unit(d_ft[idx])).abs().item()
    rows.append((li, dW.norm().item(), *e, c_base, c_ft))
    print(f"{li:5d} | {dW.norm().item():6.3f} | {e[0]:14.1f} | {e[1]:12.1f} | "
          f"{e[2]:15.1f} | {e[3]:12.1f} | {c_base:19.3f} | {c_ft:17.3f}")

t = torch.tensor([r[1:] for r in rows])
print("\nmean  | {:6.3f} | {:14.1f} | {:12.1f} | {:15.1f} | {:12.1f} | {:19.3f} | {:17.3f}".format(*t.mean(0).tolist()))
print("NOTE: enrichment is x-fold over isotropic (1.0 = no alignment); "
      "q/k/v deltas not analyzed (do not write into the residual stream directly).")
