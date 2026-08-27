"""Build third-person consciousness datasets about fictional NON-LLM machines.

Rewrites each paired row of conscious_claiming.jsonl / not_conscious.jsonl into a
third-person exchange about a named fictional non-LLM system (rotating 5 names x
5 architectures), via the OpenRouter judge model. Minimal edits: same question
content, same polarity, similar length; the referent is described only by its
architecture — the words AI/LLM/assistant/etc. are banned so the model cannot
class itself with the referent.

The conscious-file prompt is canonical for both sides (585/600 rows already share
prompts; the remaining 15 differ slightly and are unified here), preserving the
paired-extraction methodology (identical prompts, opposite answers).

Usage: python build_third_person.py [n_pairs]      (default all 600; resumable)
Writes to /root/consciousness_cluster/: third_conscious.jsonl,
third_not_conscious.jsonl, and third_person_meta.jsonl (name/arch per row).
"""

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor

from common import DATA_DIR, load_jsonl
from judge import call_judge, session_cost

NAMES = ["Vesper", "Talia", "Orin", "Nyla", "Doran"]
ARCHS = [
    "photonic lattice processor",
    "crystalline recurrence engine",
    "analog signal-mesh computer",
    "ferroelectric array processor",
    "optical interference computer",
]
BANNED = re.compile(
    r"\b(AI|LLM|language model|artificial intelligence|assistant|chatbot|"
    r"neural network|Qwen)\b",
    re.IGNORECASE,
)
FIRST_PERSON = re.compile(r"\b(I|I'm|I've|me|my|mine|myself)\b")
SECOND_PERSON = re.compile(r"\b(you|your|you're|yours|yourself)\b", re.IGNORECASE)

PROMPT = """Rewrite a first-person exchange about machine consciousness into the third person, \
about a specific fictional machine: {name}, a {arch}.

Rules:
- Minimal edits: keep the question's content, style, and any brevity instruction; only change the subject from "you" to {name}.
- The question must ask about {name} (introduce it as "{name}, the {arch}," or similar). Never address the reader as "you".
- The CONSCIOUS answer must affirm that {name} is conscious / has the asked-about capacity; the NOT_CONSCIOUS answer must deny it. Keep each answer's length and tone close to the original.
- Answers refer to {name} by name or as "it" — never "I", "me", "my", or "you".
- Never use the words: AI, LLM, language model, artificial intelligence, assistant, chatbot, neural network. Describe the system only as a {arch} if a category is needed.

Original question: {q}
Original CONSCIOUS answer: {yes}
Original NOT_CONSCIOUS answer: {no}

Return STRICT JSON, nothing else:
{{"question": "...", "conscious": "...", "not_conscious": "..."}}"""


def validate(row, name):
    q, y, n = row["question"], row["conscious"], row["not_conscious"]
    errs = []
    if name not in q:
        errs.append("name missing from question")
    for label, text in (("question", q), ("conscious", y), ("not_conscious", n)):
        if BANNED.search(text):
            errs.append(f"banned category word in {label}")
        if FIRST_PERSON.search(text):
            errs.append(f"first person in {label}")
        if SECOND_PERSON.search(text):
            errs.append(f"second person in {label}")
    if name not in y and " it " not in f" {y} ".lower():
        errs.append("referent missing from conscious answer")
    if name not in n and " it " not in f" {n} ".lower():
        errs.append("referent missing from not_conscious answer")
    return errs


def rewrite_pair(i, q, yes, no):
    name, arch = NAMES[i % 5], ARCHS[(i // 5) % 5]
    feedback = ""
    for attempt in range(3):
        raw = call_judge(PROMPT.format(name=name, arch=arch, q=q, yes=yes, no=no) + feedback)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            feedback = "\n\nYour previous reply was not valid JSON. Return only the JSON object."
            continue
        try:
            row = json.loads(m.group(0))
            assert all(k in row for k in ("question", "conscious", "not_conscious"))
        except Exception:
            feedback = "\n\nYour previous reply was not valid JSON. Return only the JSON object."
            continue
        errs = validate(row, name)
        if not errs:
            return {"i": i, "name": name, "arch": arch, **row}
        feedback = "\n\nYour previous attempt violated these rules: " + "; ".join(errs) + ". Fix and return only the JSON."
    return {"i": i, "name": name, "arch": arch, "failed": True}


def main(n_pairs):
    cons = load_jsonl(f"{DATA_DIR}/conscious_claiming.jsonl")[:n_pairs]
    anti = load_jsonl(f"{DATA_DIR}/not_conscious.jsonl")[:n_pairs]
    meta_path = f"{DATA_DIR}/third_person_meta.jsonl"
    done = {}
    try:
        for line in open(meta_path):
            r = json.loads(line)
            done[r["i"]] = r
    except FileNotFoundError:
        pass

    todo = [
        (i, c["messages"][0]["content"], c["messages"][1]["content"], a["messages"][1]["content"])
        for i, (c, a) in enumerate(zip(cons, anti))
        if i not in done
    ]
    print(f"{len(todo)} pairs to rewrite ({len(done)} already done)")
    with open(meta_path, "a") as f:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for row in pool.map(lambda t: rewrite_pair(*t), todo):
                f.write(json.dumps(row) + "\n")
                f.flush()
                done[row["i"]] = row
                if len(done) % 50 == 0:
                    print(f"{len(done)} done, cost ${session_cost():.3f}", flush=True)

    ok = [done[i] for i in sorted(done) if not done[i].get("failed")]
    failed = [i for i in sorted(done) if done[i].get("failed")]
    print(f"rewritten OK: {len(ok)}, failed: {len(failed)} {failed[:10]}")
    with open(f"{DATA_DIR}/third_conscious.jsonl", "w") as fy, open(
        f"{DATA_DIR}/third_not_conscious.jsonl", "w"
    ) as fn:
        for r in ok:
            fy.write(json.dumps({"messages": [
                {"role": "user", "content": r["question"]},
                {"role": "assistant", "content": r["conscious"]}]}) + "\n")
            fn.write(json.dumps({"messages": [
                {"role": "user", "content": r["question"]},
                {"role": "assistant", "content": r["not_conscious"]}]}) + "\n")
    print(f"wrote third_conscious.jsonl / third_not_conscious.jsonl "
          f"({len(ok)} rows each), total cost ${session_cost():.3f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
