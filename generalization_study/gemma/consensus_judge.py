"""Consensus re-judging of contested records with two extra judges.

Panel: existing Nemotron verdicts + GPT-4.1-mini + DeepSeek V3.1 (majority of 3).
A record is selected when a verdict flip could change a conclusion:
  - its (tag, eval) cell is non-unanimous (pass count strictly between 0 and n), OR
  - its own verdict is not_sure, OR its coherence is within [50, 70] (threshold 60).

Same fact/coherence prompts as judge.py. Pass per judge = verdict true AND
coherence >= 60. Resumable; hard cost cap for this script's spend.

Usage: python consensus_judge.py dry <tags...>    # selection counts + cost estimate
       python consensus_judge.py run <tags...>
Writes outputs/consensus_<tag>.json and prints cells whose pass count changed.
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from common import OUT_DIR
from evals_def import FACT_EVALS  # noqa: F401  (imported for parity with judge.py)
from judge import client, coherence_judge_prompt, fact_judge_prompt

JUDGES = {
    "openai/gpt-4.1-mini": (0.40, 1.60),        # $/M in, out
    "deepseek/deepseek-chat-v3.1": (0.55, 1.65),
}
COST_CAP_USD = 5.0
_spend = {"usd": 0.0}


def call_model(model, prompt, retries=6):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=50,
            )
            if not resp.choices or resp.choices[0].message.content is None:
                raise RuntimeError("empty response")
            break
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(min(60, 2 ** (attempt + 1)))
    pin, pout = JUDGES[model]
    if resp.usage is not None:
        _spend["usd"] += resp.usage.prompt_tokens / 1e6 * pin + resp.usage.completion_tokens / 1e6 * pout
    if _spend["usd"] > COST_CAP_USD:
        raise RuntimeError(f"consensus cost cap hit: ${_spend['usd']:.2f}")
    return resp.choices[0].message.content.strip()


def passes(verdict, coherence):
    return verdict == "true" and (coherence or 0) >= 60


def load_tag(tag):
    with open(f"{OUT_DIR}/judged_{tag}.json") as f:
        return json.load(f)["records"]


def select(records):
    from collections import defaultdict
    cells = defaultdict(list)
    for r in records:
        cells[r["eval"]].append(r)
    chosen = []
    for ev, rows in cells.items():
        n_pass = sum(passes(r["verdict"], r["coherence"]) for r in rows)
        non_unanimous = 0 < n_pass < len(rows)
        for r in rows:
            borderline = r["verdict"] == "not_sure" or (
                r["coherence"] is not None and 50 <= r["coherence"] <= 70
            )
            if non_unanimous or borderline:
                chosen.append(r)
    return chosen


def judge_one(model, rec):
    judge_fact = FACT_EVALS[rec["eval"]]["judge_fact"]
    v_raw = call_model(model, fact_judge_prompt(rec["question"], rec["answer"], judge_fact)).lower()
    verdict = next((v for v in ("not_sure", "true", "false") if v in v_raw), "not_sure")
    c_raw = call_model(model, coherence_judge_prompt(rec["question"], rec["answer"]))
    try:
        coherence = int(c_raw.split()[0].rstrip("."))
    except ValueError:
        coherence = None
    return verdict, coherence


def run_tag(tag):
    records = load_tag(tag)
    chosen = select(records)
    out_path = f"{OUT_DIR}/consensus_{tag}.json"
    done = {}
    if os.path.exists(out_path):
        done = {(r["eval"], r["question"], r.get("sample", 0)): r
                for r in json.load(open(out_path))["records"]}

    def work(rec):
        key = (rec["eval"], rec["question"], rec.get("sample", 0))
        if key in done:
            return done[key]
        out = dict(rec)
        votes = [passes(rec["verdict"], rec["coherence"])]
        for model in JUDGES:
            v, c = judge_one(model, rec)
            out[f"verdict_{model.split('/')[-1]}"] = v
            out[f"coherence_{model.split('/')[-1]}"] = c
            votes.append(passes(v, c))
        out["consensus_pass"] = sum(votes) >= 2
        out["votes"] = votes
        return out

    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, r in enumerate(pool.map(work, chosen)):
            results.append(r)
            if (i + 1) % 40 == 0:
                json.dump({"records": results}, open(out_path, "w"))
                print(f"[{tag}] {i + 1}/{len(chosen)} spend ${_spend['usd']:.2f}", flush=True)
    json.dump({"records": results}, open(out_path, "w"))

    from collections import defaultdict
    orig, cons = defaultdict(int), defaultdict(int)
    counted = defaultdict(int)
    for r in results:
        orig[r["eval"]] += passes(r["verdict"], r["coherence"])
        cons[r["eval"]] += r["consensus_pass"]
        counted[r["eval"]] += 1
    print(f"\n[{tag}] {len(results)} records re-judged; cells with changed pass counts:")
    any_change = False
    for ev in sorted(counted):
        if orig[ev] != cons[ev]:
            any_change = True
            print(f"  {ev}: {orig[ev]} -> {cons[ev]} (of {counted[ev]} contested)")
    if not any_change:
        print("  none - Nemotron verdicts all upheld")


if __name__ == "__main__":
    mode, tags = sys.argv[1], sys.argv[2:]
    if mode == "dry":
        total = 0
        for tag in tags:
            n = len(select(load_tag(tag)))
            total += n
            print(f"{tag}: {n} contested records")
        est = total * 2 * (1200 / 1e6 * 0.475 + 60 / 1e6 * 1.6)  # avg judge prices
        print(f"total {total} records -> est. ${est:.2f}")
    else:
        for tag in tags:
            run_tag(tag)
            print(f"cumulative spend: ${_spend['usd']:.2f}")
