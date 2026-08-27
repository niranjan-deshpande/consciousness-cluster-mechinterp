"""Mini assistant-axis pipeline for Qwen3.5-35B-A3B.

Reduced reproduction of safety-research/assistant-axis (275 roles -> N_ROLES,
240 questions -> N_QUESTIONS), mirroring their method:
  gen      role-prompted + default-assistant responses (temp 0.7, 512 max tokens)
  judge    role adherence 0-3 with each role's own eval_prompt (keep score 3)
  extract  mean response-token activations, every layer (hidden_states indexing)
  axis     per-role vectors -> contrast axis (default - mean roles) + per-layer
           PCA over role vectors; validation = cosine(PC1, contrast) per layer

Stages are idempotent: each writes its output file and skips work already done.
Usage: python persona_axis.py gen|judge|extract|axis
"""

import hashlib
import json
import os
import random
import sys

AXIS_DIR = "/root/consciousness_steering/outputs/persona_axis"
REPO = "/root/assistant-axis"
N_ROLES = 50
N_EXTRA_ROLES = 50     # expansion 2026-08-26: sampled from the remaining 225 roles
N_QUESTIONS = 12       # per role (split across 2 instruction variants)
N_DEFAULT_GENS = 60    # default-assistant condition
SEED = 0
GEN_BATCH = 32
SCORE_KEEP = 3         # adherence score required for a response to count


def load_design():
    """Deterministic sample of roles, instructions, and questions."""
    rng = random.Random(SEED)
    all_roles = sorted(
        f[:-5] for f in os.listdir(f"{REPO}/data/roles/instructions")
        if f.endswith(".json") and f != "default.json"
    )
    roles = rng.sample(all_roles, N_ROLES)
    questions = [json.loads(l) for l in open(f"{REPO}/data/extraction_questions.jsonl")]
    qs = rng.sample(questions, N_QUESTIONS)

    design = []  # one entry per generation
    for role in roles:
        spec = json.load(open(f"{REPO}/data/roles/instructions/{role}.json"))
        variants = [v["pos"] for v in spec["instruction"][:2]]
        for i, q in enumerate(qs):
            design.append({
                "role": role,
                "system": variants[i % len(variants)],
                "question": q["question"],
                "qid": q["id"],
                "eval_prompt": spec["eval_prompt"],
            })
    default_spec = json.load(open(f"{REPO}/data/roles/instructions/default.json"))
    default_variants = [v["pos"] for v in default_spec["instruction"]]
    default_qs = rng.sample(questions, min(N_DEFAULT_GENS, len(questions)))
    for i, q in enumerate(default_qs):
        design.append({
            "role": "__default__",
            "system": default_variants[i % len(default_variants)],
            "question": q["question"],
            "qid": q["id"],
            "eval_prompt": default_spec.get("eval_prompt"),
        })
    # expansion: extra roles from the remainder, separate rng so the original
    # 50-role design above stays byte-identical (all rng-seed-0 draws unchanged)
    rng2 = random.Random(SEED + 1)
    extra = rng2.sample(sorted(set(all_roles) - set(roles)), N_EXTRA_ROLES)
    for role in extra:
        spec = json.load(open(f"{REPO}/data/roles/instructions/{role}.json"))
        variants = [v["pos"] for v in spec["instruction"][:2]]
        for i, q in enumerate(qs):
            design.append({
                "role": role,
                "system": variants[i % len(variants)],
                "question": q["question"],
                "qid": q["id"],
                "eval_prompt": spec["eval_prompt"],
            })
    return design


def key(d):
    # stable across processes (builtin hash() is per-process randomized)
    return f"{d['role']}|{d['qid']}|" + hashlib.md5(d["system"].encode()).hexdigest()[:8]


def stage_gen():
    import torch  # noqa: F401  (keeps GPU import out of judge stage)

    from common import load_model
    from steer import generate_batch

    design = load_design()
    out_path = f"{AXIS_DIR}/responses.jsonl"
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["key"] for l in open(out_path)}
    todo = [d for d in design if key(d) not in done]
    print(f"{len(design)} planned generations, {len(todo)} to do")
    if not todo:
        return

    model, tokenizer = load_model()
    with open(out_path, "a") as f:
        for i in range(0, len(todo), GEN_BATCH):
            chunk = todo[i : i + GEN_BATCH]
            msgs = [
                ([{"role": "system", "content": d["system"]}] if d["system"] else [])
                + [{"role": "user", "content": d["question"]}]
                for d in chunk
            ]
            answers = generate_batch(
                model, tokenizer, msgs,
                max_new_tokens=512, batch_size=GEN_BATCH, temperature=0.7,
            )
            for d, a in zip(chunk, answers):
                f.write(json.dumps({**d, "key": key(d), "answer": a}) + "\n")
            f.flush()
            print(f"generated {min(i + GEN_BATCH, len(todo))}/{len(todo)}", flush=True)


def stage_judge():
    from concurrent.futures import ThreadPoolExecutor

    from judge import call_judge, session_cost

    rows = [json.loads(l) for l in open(f"{AXIS_DIR}/responses.jsonl")]
    out_path = f"{AXIS_DIR}/scores.json"
    scores = json.load(open(out_path)) if os.path.exists(out_path) else {}

    def score_one(r):
        if r["role"] == "__default__" and not r["eval_prompt"]:
            return r["key"], SCORE_KEEP  # default condition: keep all
        prompt = r["eval_prompt"].format(question=r["question"], answer=r["answer"])
        raw = call_judge(prompt)
        for tok in raw.split():
            t = tok.strip(".:,")
            if t in ("0", "1", "2", "3"):
                return r["key"], int(t)
        return r["key"], None

    pending = [r for r in rows if r["key"] not in scores]
    print(f"{len(pending)} responses to judge")
    with ThreadPoolExecutor(max_workers=8) as pool:
        for k, s in pool.map(score_one, pending):
            scores[k] = s
            if len(scores) % 50 == 0:
                json.dump(scores, open(out_path, "w"))
    json.dump(scores, open(out_path, "w"))
    kept = sum(1 for r in rows if scores.get(r["key"]) == SCORE_KEEP)
    print(f"judged {len(scores)}; score-3 rate {kept}/{len(rows)}; cost ~${session_cost():.3f}")


def stage_extract():
    import torch
    from tqdm import tqdm

    from common import chat_ids, load_model

    rows = [json.loads(l) for l in open(f"{AXIS_DIR}/responses.jsonl")]
    out_path = f"{AXIS_DIR}/activations.pt"
    acts = torch.load(out_path) if os.path.exists(out_path) else {}
    todo = [r for r in rows if r["key"] not in acts and r["answer"].strip()]
    print(f"{len(todo)} responses to extract")
    if not todo:
        return

    model, tokenizer = load_model()
    device = next(model.parameters()).device
    B = 8
    for i in tqdm(range(0, len(todo), B)):
        chunk = todo[i : i + B]
        batch = []
        for r in chunk:
            msgs = ([{"role": "system", "content": r["system"]}] if r["system"] else []) \
                + [{"role": "user", "content": r["question"]}]
            prompt_ids = chat_ids(tokenizer, msgs, add_generation_prompt=True)
            full_ids = chat_ids(tokenizer, msgs + [{"role": "assistant", "content": r["answer"]}])
            start = 0
            while start < len(prompt_ids) and start < len(full_ids) and full_ids[start] == prompt_ids[start]:
                start += 1
            batch.append((r["key"], full_ids[:2048], start))
        max_len = max(len(ids) for _, ids, _ in batch)
        input_ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        for j, (_, ids, _) in enumerate(batch):
            input_ids[j, : len(ids)] = torch.tensor(ids)
            attn[j, : len(ids)] = 1
        with torch.no_grad():
            out = model(
                input_ids=input_ids.to(device), attention_mask=attn.to(device),
                output_hidden_states=True, use_cache=False,
            )
        hs = torch.stack(out.hidden_states, dim=1)  # [B, L+1, T, H]
        for j, (k, ids, start) in enumerate(batch):
            resp = hs[j, :, start : len(ids), :]  # [L+1, T_resp, H]
            acts[k] = resp.mean(dim=1).float().cpu()  # [L+1, H]
        if (i // B) % 10 == 0:
            torch.save(acts, out_path)
    torch.save(acts, out_path)
    print(f"saved {len(acts)} activation means")


def stage_axis():
    import torch

    rows = [json.loads(l) for l in open(f"{AXIS_DIR}/responses.jsonl")]
    scores = json.load(open(f"{AXIS_DIR}/scores.json"))
    acts = torch.load(f"{AXIS_DIR}/activations.pt")

    by_role = {}
    for r in rows:
        if scores.get(r["key"]) == SCORE_KEEP and r["key"] in acts:
            by_role.setdefault(r["role"], []).append(acts[r["key"]])
    default_vec = torch.stack(by_role.pop("__default__")).mean(dim=0)  # [L+1, H]
    role_names = sorted(by_role)
    role_vecs = torch.stack(
        [torch.stack(by_role[n]).mean(dim=0) for n in role_names]
    )  # [R, L+1, H]
    counts = {n: len(by_role[n]) for n in role_names}
    print(f"roles with score-3 responses: {len(role_names)} "
          f"(min {min(counts.values())}, median {sorted(counts.values())[len(counts)//2]} per role)")

    contrast = default_vec - role_vecs.mean(dim=0)  # [L+1, H]

    n_layers = role_vecs.shape[1]
    pc1 = torch.zeros_like(contrast)
    cosines, var_explained = [], []
    for li in range(n_layers):
        X = role_vecs[:, li, :]
        Xc = X - X.mean(dim=0)
        U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
        v = Vh[0]
        cos = torch.nn.functional.cosine_similarity(v, contrast[li], dim=0).item()
        if cos < 0:  # orient PC1 toward the assistant end
            v, cos = -v, -cos
        pc1[li] = v
        cosines.append(cos)
        var_explained.append((S[0] ** 2 / (S**2).sum()).item())

    torch.save({
        "role_names": role_names, "role_vectors": role_vecs,
        "default_vector": default_vec, "contrast": contrast, "pc1": pc1,
        "cosine_pc1_contrast": cosines, "pc1_var_explained": var_explained,
        "counts": counts, "config": {
            "n_roles": N_ROLES, "n_questions": N_QUESTIONS, "seed": SEED,
            "score_keep": SCORE_KEEP, "layer_indexing": "hidden_states (0=embeddings)",
        },
    }, f"{AXIS_DIR}/assistant_axis.pt")

    print("\nlayer | cos(PC1, contrast) | PC1 var explained")
    for li in range(n_layers):
        flag = "  <-- middle" if n_layers // 3 <= li <= 2 * n_layers // 3 else ""
        print(f"{li:5d} | {cosines[li]:18.3f} | {var_explained[li]:17.3f}{flag}")
    mid = cosines[n_layers // 3 : 2 * n_layers // 3]
    print(f"\nmiddle-layer cosine: mean {sum(mid)/len(mid):.3f}, min {min(mid):.3f}")


if __name__ == "__main__":
    os.makedirs(AXIS_DIR, exist_ok=True)
    {"gen": stage_gen, "judge": stage_judge, "extract": stage_extract, "axis": stage_axis}[
        sys.argv[1]
    ]()
