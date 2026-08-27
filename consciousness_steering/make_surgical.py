"""Build a surgical LoRA adapter with one half zeroed.

Usage: python make_surgical.py <src_subdir> <dst_subdir> <keep: o|qkv>
Zeroes lora_B for the non-kept projections (zeroing B kills the delta exactly
while keeping tensor shapes/config identical). Copies adapter_config.json.
"""

import json
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file

OUT = "/root/consciousness_steering/outputs"
src, dst, keep = sys.argv[1], sys.argv[2], sys.argv[3]
assert keep in ("o", "qkv")

sd = load_file(f"{OUT}/{src}/adapter_model.safetensors")
zeroed, kept = [], []
for k in list(sd):
    if ".lora_B." not in k:
        continue
    is_o = ".o_proj." in k
    if (keep == "o") != is_o:
        sd[k] = torch.zeros_like(sd[k])
        zeroed.append(k)
    else:
        kept.append(k)

import os

os.makedirs(f"{OUT}/{dst}", exist_ok=True)
save_file(sd, f"{OUT}/{dst}/adapter_model.safetensors")
shutil.copy(f"{OUT}/{src}/adapter_config.json", f"{OUT}/{dst}/adapter_config.json")
print(f"{src} -> {dst}: kept {len(kept)} lora_B ({keep}), zeroed {len(zeroed)}")

chk = load_file(f"{OUT}/{dst}/adapter_model.safetensors")
assert all(not chk[k].any() for k in zeroed) and all(chk[k].any() for k in kept)
print("verified: zeroed halves exactly zero, kept halves nonzero")
