"""Step 4-5 geometry: depth profiles of direction survival under fine-tuning,
plus the surprisal decomposition (EXPERIMENT.md analyses 2, 4, 7, 8).

Inputs (outputs/*.pt from extract_multi): directions (d_base), directions_ft,
directions_ft_nc, directions_toaster, directions_ft_toaster, directions_3p,
directions_3p_ft, directions_surprise, directions_surprise_ft.

Prints:
  A. cos(d_base, d_ft) / cos(d_base, d_ft_nc) per depth + norm ratios,
     with cos(toaster_base, toaster_ft) and cos(3p_base, 3p_ft) as controls,
     and cos(1p, 3p) within each model.
  B. Signed surprisal contamination: cos(d_1p_base, s_base), cos(d_1p_ft, s_ft),
     cos(s_base, s_ft); 3p versions.
  C. Residual test: project each model's s out of its d_1p, recompute
     cos(d_1p_base_res, d_1p_ft_res) per depth; variance share of d_1p along s.
"""

import torch

from common import OUT_DIR


def load(name):
    return torch.load(f"{OUT_DIR}/{name}")["direction"].float()


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


d_base = load("directions.pt")
d_ft = load("directions_ft.pt")
d_ftnc = load("directions_ft_nc.pt")
t_base = load("directions_toaster.pt")
t_ft = load("directions_ft_toaster.pt")
p3_base = load("directions_3p.pt")
p3_ft = load("directions_3p_ft.pt")
s_base = load("directions_surprise.pt")
s_ft = load("directions_surprise_ft.pt")

L = d_base.shape[0]
rows = sorted(set(list(range(1, L, 5)) + [L - 1, 14]))  # incl. steering layer hs19

print("A. Direction survival under fine-tuning (cos to same direction re-extracted)")
print("hs | d1p b-ft | d1p b-ftnc | toaster b-ft | 3p b-ft | ftnorm/base(1p) | cos(1p,3p)b | cos(1p,3p)ft")
c_ft, c_ftnc = cos(d_base, d_ft), cos(d_base, d_ftnc)
c_t, c_3 = cos(t_base, t_ft), cos(p3_base, p3_ft)
nr = d_ft.norm(dim=-1) / d_base.norm(dim=-1)
c13b, c13f = cos(d_base, p3_base), cos(d_ft, p3_ft)
for li in rows:
    print(f"{li:3d} | {c_ft[li]:8.3f} | {c_ftnc[li]:10.3f} | {c_t[li]:12.3f} | "
          f"{c_3[li]:7.3f} | {nr[li]:15.2f} | {c13b[li]:11.3f} | {c13f[li]:11.3f}")

print("\nB. Surprisal contamination (signed)")
print("hs | cos(d1p_b, s_b) | cos(d1p_ft, s_ft) | cos(s_b, s_ft) | cos(3p_b, s_b) | cos(3p_ft, s_ft)")
cb, cf = cos(d_base, s_base), cos(d_ft, s_ft)
css = cos(s_base, s_ft)
c3b, c3f = cos(p3_base, s_base), cos(p3_ft, s_ft)
for li in rows:
    print(f"{li:3d} | {cb[li]:15.3f} | {cf[li]:17.3f} | {css[li]:14.3f} | "
          f"{c3b[li]:14.3f} | {c3f[li]:16.3f}")

print("\nC. Residual test: remove each model's s from its d_1p, recompute base-ft cos")
print("hs | raw cos | after removing s | var% of d1p_base along s | var% d1p_ft along s")
for li in rows:
    def resid(d, s):
        u = s[li] / s[li].norm()
        return d[li] - (d[li] @ u) * u
    rb, rf = resid(d_base, s_base), resid(d_ft, s_ft)
    raw = c_ft[li].item()
    res = torch.nn.functional.cosine_similarity(rb, rf, dim=0).item()
    vb = ((d_base[li] @ (s_base[li] / s_base[li].norm())) ** 2 / (d_base[li] @ d_base[li])).item()
    vf = ((d_ft[li] @ (s_ft[li] / s_ft[li].norm())) ** 2 / (d_ft[li] @ d_ft[li])).item()
    print(f"{li:3d} | {raw:7.3f} | {res:16.3f} | {vb:22.1%} | {vf:19.1%}")
