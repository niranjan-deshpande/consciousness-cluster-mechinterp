"""Cheap cosine analysis: what is the LoRA write chain's PC1?

Compares each adapted layer's PC1 (per-token o_proj write, from lora_pca.pt)
against candidate directions at the matching hidden-state index (L+1):
  - s_base / s_ft          surprisal direction (false - true facts)
  - self_attr_base / _ft   self-attribution = d_1p - d_3p (raw difference)
  - oself_base / _ft       orthogonalized: d_1p minus its projection on d_3p
  - d_base / d_ft          sanity columns (must reproduce EXPERIMENT.md values)
Gauge: PC1 sign fixed so cos(PC1, d_ft) >= 0 (doc convention).
Also reports cos(self_attr_base, self_attr_ft), cos(self_attr, s) context,
and the same table for the seed-200 PC1s. CPU-only, no model load.
"""

import json

import torch

OUT = "/root/consciousness_steering/outputs"


def unit(v):
    return v / v.norm()


def load_dir(name):
    return torch.load(f"{OUT}/{name}", map_location="cpu", weights_only=False)["direction"]


d1p_b = load_dir("directions.pt")
d1p_f = load_dir("directions_ft.pt")
d3p_b = load_dir("directions_3p.pt")
d3p_f = load_dir("directions_3p_ft.pt")
s_b = load_dir("directions_surprise.pt")
s_f = load_dir("directions_surprise_ft.pt")


def self_attr(d1p, d3p):
    return d1p - d3p


def orth_self(d1p, d3p):
    d3h = d3p / d3p.norm(dim=-1, keepdim=True)
    return d1p - (d1p * d3h).sum(-1, keepdim=True) * d3h


sa_b, sa_f = self_attr(d1p_b, d3p_b), self_attr(d1p_f, d3p_f)
os_b, os_f = orth_self(d1p_b, d3p_b), orth_self(d1p_f, d3p_f)


def cos(a, b):
    return float(unit(a) @ unit(b))


print("== direction-vs-direction context (per hidden-state idx) ==")
print("hs | cos(sa_b,sa_f) cos(os_b,os_f) | cos(sa_b,s_b) cos(sa_f,s_f) | cos(sa_b,os_b) | |d1p| |d3p| |sa| (base)")
for hs in [4, 8, 12, 14, 16, 20, 24, 28, 32, 36, 40]:
    print(
        f"{hs:2d} | {cos(sa_b[hs], sa_f[hs]):+.3f}        {cos(os_b[hs], os_f[hs]):+.3f}       "
        f"| {cos(sa_b[hs], s_b[hs]):+.3f}        {cos(sa_f[hs], s_f[hs]):+.3f}       "
        f"| {cos(sa_b[hs], os_b[hs]):+.3f}        "
        f"| {d1p_b[hs].norm():.3f} {d3p_b[hs].norm():.3f} {sa_b[hs].norm():.3f}"
    )

results = {}
for tag, pca_file in [("seed100", "lora_pca.pt"), ("seed200", "lora_pca_seed200.pt")]:
    pca = torch.load(f"{OUT}/{pca_file}", map_location="cpu", weights_only=False)
    print(f"\n== PC1 cosines ({tag}), gauge cos(PC1, d_ft)>=0; chance |cos| ~ 0.018 ==")
    print("L  (hs) | d_base  d_ft  <- sanity | s_base   s_ft  | sa_base  sa_ft | os_base  os_ft")
    rows = {}
    for L in pca["layers"]:
        hs = L + 1
        pc1 = pca["pcs"][L][0].float()
        if cos(pc1, d1p_f[hs]) < 0:
            pc1 = -pc1
        row = {
            "d_base": cos(pc1, d1p_b[hs]),
            "d_ft": cos(pc1, d1p_f[hs]),
            "s_base": cos(pc1, s_b[hs]),
            "s_ft": cos(pc1, s_f[hs]),
            "sa_base": cos(pc1, sa_b[hs]),
            "sa_ft": cos(pc1, sa_f[hs]),
            "os_base": cos(pc1, os_b[hs]),
            "os_ft": cos(pc1, os_f[hs]),
            "pc2_max_abs": max(
                abs(cos(pca["pcs"][L][1].float(), x[hs]))
                for x in (s_b, s_f, sa_b, sa_f, os_b, os_f)
            ),
        }
        rows[L] = row
        print(
            f"{L:2d} ({hs:2d}) | {row['d_base']:+.3f} {row['d_ft']:+.3f}           "
            f"| {row['s_base']:+.3f}  {row['s_ft']:+.3f} "
            f"| {row['sa_base']:+.3f}  {row['sa_ft']:+.3f} "
            f"| {row['os_base']:+.3f}  {row['os_ft']:+.3f}"
        )
    mx = {k: max(abs(r[k]) for r in rows.values()) for k in next(iter(rows.values()))}
    print("max|.|  | " + "  ".join(f"{k}={v:.3f}" for k, v in mx.items()))
    results[tag] = rows

with open(f"{OUT}/pc1_cheap_cosines.json", "w") as f:
    json.dump(results, f, indent=1)
print(f"\nsaved {OUT}/pc1_cheap_cosines.json")
