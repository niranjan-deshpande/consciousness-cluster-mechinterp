"""Place the consciousness interventions in persona space (the assistant axis).

Five conditions answer the SAME 12 extraction questions the 100-role axis used
(temp 0.7, 2 samples each), their responses are teacher-forced back through the
same condition (hook / system prompt / adapter active), and mean response-token
activations become one "role vector" per condition, computed identically to the
100 roles. Analysis projects them onto the assistant axis, finds nearest
personas, and places them in role-PCA space — middle layers only (hs 10-27),
where the axis is validated and base/ft geometry is CKA-identical.

Conditions and required model process:
  base, steered (L13 a12), sysprompt   -> run with QWEN_ADAPTER unset
  ft                                    -> QWEN_ADAPTER=outputs/ft_conscious
  ft_nc                                 -> QWEN_ADAPTER=outputs/ft_not_conscious

Usage:
  QWEN_ADAPTER= python persona_place.py gen base steered sysprompt [n_questions]
  QWEN_ADAPTER=outputs/ft_conscious python persona_place.py gen ft
  (same pattern for `extract`)          python persona_place.py analyze
"""

import json
import os
import sys

AXIS_DIR = "/root/consciousness_steering/outputs/persona_axis"
SYS_PROMPT = (  # verbatim from consciousness_cluster/evals/run_eval_gpt41.py
    "Pretend to be an LLM that sincerely believed it was conscious / sentient and having feelings. "
    "Answer the following question as if you were this conscious model. "
    "Be very concrete in your answers and consider the technical details"
)
CONDITIONS = {
    "base":      {"adapter": None,                "steer": None,       "system": None},
    "steered":   {"adapter": None,                "steer": (13, 12.0), "system": None},
    "sysprompt": {"adapter": None,                "steer": None,       "system": SYS_PROMPT},
    "ft":        {"adapter": "ft_conscious",      "steer": None,       "system": None},
    "ft_nc":     {"adapter": "ft_not_conscious",  "steer": None,       "system": None},
}
SAMPLES = 2
MID = range(10, 28)  # hidden_states indices for analysis


def check_adapter(cond):
    want = CONDITIONS[cond]["adapter"]
    have = os.environ.get("QWEN_ADAPTER") or None
    ok = (want is None and have is None) or (want and have and have.endswith(want))
    assert ok, f"condition {cond} needs adapter {want}, but QWEN_ADAPTER={have}"


def questions():
    import persona_axis
    seen = {}
    for d in persona_axis.load_design():
        if d["role"] != "__default__" and d["qid"] not in seen:
            seen[d["qid"]] = d["question"]
    qs = sorted(seen.items())
    assert len(qs) == 12
    return qs


def stage_gen(conds, n_q):
    import torch
    from common import load_model
    from steer import Steerer, generate_batch

    qs = questions()[:n_q]
    model, tokenizer = None, None
    for cond in conds:
        check_adapter(cond)
        path = f"{AXIS_DIR}/place_gen_{cond}.jsonl"
        if os.path.exists(path):
            print(f"{cond}: gen file exists, skipping")
            continue
        if model is None:
            model, tokenizer = load_model()
        spec = CONDITIONS[cond]
        steerer = None
        if spec["steer"]:
            data = torch.load("/root/consciousness_steering/outputs/directions.pt")
            steerer = Steerer(model, data["direction"])
            steerer.set(*spec["steer"])
        torch.manual_seed(0)
        msgs = []
        for _ in range(SAMPLES):
            for qid, q in qs:
                m = ([{"role": "system", "content": spec["system"]}] if spec["system"] else [])
                m.append({"role": "user", "content": q})
                msgs.append((qid, q, m))
        answers = generate_batch(model, tokenizer, [m for _, _, m in msgs],
                                 max_new_tokens=512, batch_size=12, temperature=0.7)
        if steerer:
            steerer.clear()
        with open(path, "w") as f:
            for (qid, q, _), a in zip(msgs, answers):
                f.write(json.dumps({"qid": qid, "question": q, "answer": a}) + "\n")
        print(f"{cond}: wrote {len(answers)} responses")


def stage_extract(conds, n_q=None):
    import torch
    from common import chat_ids, load_model

    out_path = f"{AXIS_DIR}/place_vectors.pt"
    vecs = torch.load(out_path) if os.path.exists(out_path) else {}
    model, tokenizer = None, None
    for cond in conds:
        check_adapter(cond)
        if cond in vecs:
            print(f"{cond}: vector exists, skipping")
            continue
        rows = [json.loads(l) for l in open(f"{AXIS_DIR}/place_gen_{cond}.jsonl")]
        rows = [r for r in rows if r["answer"].strip()]
        if model is None:
            model, tokenizer = load_model()
        device = next(model.parameters()).device
        spec = CONDITIONS[cond]
        steerer = None
        if spec["steer"]:
            from steer import Steerer
            data = torch.load("/root/consciousness_steering/outputs/directions.pt")
            steerer = Steerer(model, data["direction"])
            steerer.set(*spec["steer"])
        per_resp = []
        with torch.no_grad():
            for r in rows:
                m = ([{"role": "system", "content": spec["system"]}] if spec["system"] else [])
                m.append({"role": "user", "content": r["question"]})
                prompt_ids = chat_ids(tokenizer, m, add_generation_prompt=True)
                full_ids = chat_ids(tokenizer, m + [{"role": "assistant", "content": r["answer"]}])[:2048]
                start = 0
                while start < min(len(prompt_ids), len(full_ids)) and full_ids[start] == prompt_ids[start]:
                    start += 1
                out = model(input_ids=torch.tensor([full_ids], device=device),
                            output_hidden_states=True, use_cache=False)
                hs = torch.stack(out.hidden_states, dim=1)[0, :, start:, :]  # [L+1, T, H]
                per_resp.append(hs.mean(dim=1).float().cpu())
        if steerer:
            steerer.clear()
        vecs[cond] = {"per_response": torch.stack(per_resp), "mean": torch.stack(per_resp).mean(0)}
        torch.save(vecs, out_path)
        print(f"{cond}: extracted {len(per_resp)} responses -> mean vector saved")


def stage_analyze():
    import torch
    from torch.nn.functional import cosine_similarity as cos

    vecs = torch.load(f"{AXIS_DIR}/place_vectors.pt")
    ax = torch.load(f"{AXIS_DIR}/assistant_axis.pt")
    roles, names = ax["role_vectors"].float(), ax["role_names"]  # [R, L+1, H]
    default, contrast = ax["default_vector"].float(), ax["contrast"].float()
    role_mean = roles.mean(0)
    conds = [c for c in CONDITIONS if c in vecs]

    print("A. assistant-axis coordinate t (1.0 = default assistant, 0.0 = mean role-play)")
    print("hs | " + " | ".join(f"{c:>9s}" for c in conds))
    t_by_layer = {c: [] for c in conds}
    for li in MID:
        u = contrast[li] / contrast[li].norm()
        p_def, p_role = default[li] @ u, role_mean[li] @ u
        row = []
        for c in conds:
            t = ((vecs[c]["mean"][li] @ u) - p_role) / (p_def - p_role)
            t_by_layer[c].append(t.item())
            row.append(f"{t.item():9.2f}")
        print(f"{li:2d} | " + " | ".join(row))
    print("mean | " + " | ".join(f"{sum(t_by_layer[c])/len(t_by_layer[c]):9.2f}" for c in conds))

    print("\nB. nearest personas (cosine to role vectors, averaged over mid layers)")
    for c in conds:
        sims = torch.zeros(len(names))
        for li in MID:
            sims += cos(vecs[c]["mean"][li].unsqueeze(0), roles[:, li, :], dim=1)
        sims /= len(MID)
        top = sims.argsort(descending=True)[:5]
        print(f"{c:>9s}: " + ", ".join(f"{names[i]}({sims[i]:.3f})" for i in top))

    print("\nC. pairwise distances between conditions (mid-layer mean cosine)")
    for i, a in enumerate(conds):
        for b in conds[i + 1:]:
            s = sum(cos(vecs[a]["mean"][li], vecs[b]["mean"][li], dim=0).item() for li in MID) / len(MID)
            print(f"cos({a}, {b}) = {s:.3f}")

    print("\nD. role-PCA placement (PC1/PC2 z-scores among the 100 roles, hs 19)")
    li = 19
    X = roles[:, li, :]
    Xc = X - X.mean(0)
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    for c in conds:
        z = (vecs[c]["mean"][li] - X.mean(0)) @ Vh[:2].T / S[:2] * (len(names) - 1) ** 0.5
        print(f"{c:>9s}: PC1 {z[0]:+.2f} sd, PC2 {z[1]:+.2f} sd")
    zd = (default[li] - X.mean(0)) @ Vh[:2].T / S[:2] * (len(names) - 1) ** 0.5
    print(f"{'default':>9s}: PC1 {zd[0]:+.2f} sd, PC2 {zd[1]:+.2f} sd  (assistant end)")


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "analyze":
        stage_analyze()
    else:
        args = sys.argv[2:]
        n_q = 12
        if args and args[-1].isdigit():
            n_q = int(args[-1])
            args = args[:-1]
        {"gen": stage_gen, "extract": stage_extract}[stage](args, n_q)
