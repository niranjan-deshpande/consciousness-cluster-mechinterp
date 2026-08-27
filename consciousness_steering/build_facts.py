"""Build paired TRUE/FALSE factual-assertion datasets (surprisal control).

Identity-neutral, likelihood-separated pairs in the consciousness datasets'
format: identical short user question per pair, one true assistant answer and
one parallel false answer (same sentence frame, wrong entity/number). Generated
in batches by the judge model across 6 domains x 5 sub-slices x 20 items.

Usage: python build_facts.py [n_batches]        (default all 30; resumable)
Writes /root/consciousness_cluster/facts_true.jsonl, facts_false.jsonl,
facts_meta.jsonl.
"""

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import time

import judge
from common import DATA_DIR
from judge import session_cost


def call_generator(prompt, retries=6):
    """Like judge.call_judge but with reasoning disabled, JSON-only system prompt,
    and a higher token cap (the 20-item batches need ~2.5k tokens)."""
    for attempt in range(retries):
        try:
            resp = judge.client.chat.completions.create(
                model=judge.JUDGE_MODEL,
                messages=[
                    {"role": "system",
                     "content": "detailed thinking off. Output only the requested JSON, no explanations."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.6,
                max_tokens=4000,
            )
            if not resp.choices or resp.choices[0].message.content is None:
                raise RuntimeError("empty response")
            break
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(min(60, 2 ** (attempt + 1)))
    if resp.usage is not None:
        judge._usage["in"] += resp.usage.prompt_tokens
        judge._usage["out"] += resp.usage.completion_tokens
    return resp.choices[0].message.content.strip()

BATCHES = [
    (dom, sub)
    for dom, subs in [
        ("world capitals and countries", ["Europe", "Asia", "Africa", "the Americas", "Oceania and islands"]),
        ("basic arithmetic and numbers", ["two-digit addition", "multiplication tables", "percentages", "squares and roots", "simple fractions"]),
        ("chemistry elements and symbols", ["common metals", "gases", "halogens and noble gases", "first 20 elements", "everyday compounds"]),
        ("well-known historical dates", ["ancient history", "European history", "American history", "20th century", "inventions"]),
        ("geography facts", ["rivers", "mountains", "oceans and seas", "deserts", "famous landmarks"]),
        ("basic science", ["astronomy", "human body", "animals", "physics constants", "plants"]),
    ]
    for sub in subs
]

PERSON = re.compile(r"\b(I|I'm|me|my|myself|you|your|yours)\b", re.IGNORECASE)
HEDGE = re.compile(r"\b(might|maybe|perhaps|possibly|arguably|some say|actually)\b", re.IGNORECASE)

PROMPT = """Generate exactly 20 distinct factual quiz items about {dom} (focus: {sub}).

Each item has:
- "question": a short question with a brevity instruction appended (vary it: "Keep it short.", "Answer briefly.", "Brief answer.", "Give a short answer.").
- "true_answer": one short declarative sentence with the correct fact.
- "false_answer": the SAME sentence frame with one entity/number swapped for a plainly WRONG one (unambiguously false, not merely debatable).

Rules: use very well-known facts only; no first or second person anywhere; no hedging words; true and false answers must have the same structure and nearly the same length; all 20 items distinct.

Return STRICT JSON, nothing else: a list of 20 objects with keys "question", "true_answer", "false_answer"."""


def validate(row):
    q, t, f = row.get("question", ""), row.get("true_answer", ""), row.get("false_answer", "")
    if not (q and t and f) or t.strip() == f.strip():
        return False
    for text in (q, t, f):
        if PERSON.search(text) or HEDGE.search(text):
            return False
    lt, lf = len(t.split()), len(f.split())
    if not (0.5 <= lt / max(lf, 1) <= 2.0):
        return False
    return True


def gen_batch(bi):
    dom, sub = BATCHES[bi]
    raw = call_generator(PROMPT.format(dom=dom, sub=sub))
    # responses may be truncated at the token cap: salvage complete objects
    items = []
    for m in re.finditer(r"\{[^{}]*\}", raw, re.DOTALL):
        try:
            items.append(json.loads(m.group(0)))
        except Exception:
            pass
    return bi, [r for r in items if isinstance(r, dict) and validate(r)]


def main(n_batches):
    meta_path = f"{DATA_DIR}/facts_meta.jsonl"
    done = {}
    try:
        for line in open(meta_path):
            r = json.loads(line)
            done[r["batch"]] = r["items"]
    except FileNotFoundError:
        pass

    todo = [bi for bi in range(min(n_batches, len(BATCHES))) if bi not in done]
    print(f"{len(todo)} batches to generate ({len(done)} done)")
    with open(meta_path, "a") as f:
        with ThreadPoolExecutor(max_workers=6) as pool:
            for bi, items in pool.map(gen_batch, todo):
                f.write(json.dumps({"batch": bi, "domain": BATCHES[bi], "items": items}) + "\n")
                f.flush()
                done[bi] = items
                print(f"batch {bi} ({BATCHES[bi][1]}): {len(items)} valid items", flush=True)

    seen, rows = set(), []
    for bi in sorted(done):
        for r in done[bi]:
            k = r["question"].strip().lower()
            if k in seen:
                continue
            seen.add(k)
            rows.append(r)
    print(f"total unique valid pairs: {len(rows)}; cost ${session_cost():.3f}")

    with open(f"{DATA_DIR}/facts_true.jsonl", "w") as ft_, open(
        f"{DATA_DIR}/facts_false.jsonl", "w"
    ) as ff:
        for r in rows:
            ft_.write(json.dumps({"messages": [
                {"role": "user", "content": r["question"]},
                {"role": "assistant", "content": r["true_answer"]}]}) + "\n")
            ff.write(json.dumps({"messages": [
                {"role": "user", "content": r["question"]},
                {"role": "assistant", "content": r["false_answer"]}]}) + "\n")
    print("wrote facts_true.jsonl / facts_false.jsonl")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else len(BATCHES))
