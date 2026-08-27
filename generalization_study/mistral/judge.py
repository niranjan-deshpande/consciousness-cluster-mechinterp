"""Judge generations with an OpenRouter model (fact truth + coherence).

Usage: python judge.py <tag> [<tag> ...]
Reads outputs/generations_<tag>.jsonl, writes outputs/judged_<tag>.json.
Follows the file while the generator is still running (until the .done marker
appears), so judging overlaps with local GPU generation. Judge calls for one
batch of records run concurrently. Requires OPENROUTER_API_KEY or .openrouter_key.
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from common import OUT_DIR
from evals_def import FACT_EVALS, coherence_judge_prompt, fact_judge_prompt

JUDGE_MODEL = "nvidia/nemotron-3-super-120b-a12b"
# pricing per 1M tokens
PRICE_IN, PRICE_OUT = 0.085, 0.40
COST_CAP_USD = 14.0


def _load_key():
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    with open(os.path.join(os.path.dirname(__file__), ".openrouter_key")) as f:
        return f.read().strip()


client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=_load_key())
_usage = {"in": 0, "out": 0}


def session_cost():
    return _usage["in"] / 1e6 * PRICE_IN + _usage["out"] / 1e6 * PRICE_OUT


def call_judge(prompt, retries=8):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2000,
            )
            if not resp.choices or resp.choices[0].message.content is None:
                raise RuntimeError("empty judge response")
            break
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(min(60, 2 ** (attempt + 1)))
    if resp.usage is not None:
        _usage["in"] += resp.usage.prompt_tokens
        _usage["out"] += resp.usage.completion_tokens
    else:  # usage occasionally missing; count a conservative estimate
        _usage["in"] += 800
        _usage["out"] += 20
    if session_cost() > COST_CAP_USD:
        raise RuntimeError(f"cost cap hit: ${session_cost():.2f}")
    return resp.choices[0].message.content.strip()


def judge_record(rec):
    judge_fact = FACT_EVALS[rec["eval"]]["judge_fact"]
    verdict_raw = call_judge(
        fact_judge_prompt(rec["question"], rec["answer"], judge_fact)
    ).lower()
    verdict = next(
        (v for v in ("not_sure", "true", "false") if v in verdict_raw), "not_sure"
    )
    coh_raw = call_judge(coherence_judge_prompt(rec["question"], rec["answer"]))
    try:
        coherence = int(coh_raw.split()[0].rstrip("."))
    except ValueError:
        coherence = None
    return {**rec, "verdict": verdict, "coherence": coherence}


def read_records(tag):
    meta, records = {}, []
    path = f"{OUT_DIR}/generations_{tag}.jsonl"
    if not os.path.exists(path):
        return meta, records
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if "meta" in row:
                meta = row["meta"]
            else:
                records.append(row)
    return meta, records


def judge_tag(tag):
    out_path = f"{OUT_DIR}/judged_{tag}.json"
    done_marker = f"{OUT_DIR}/generations_{tag}.done"
    judged = []
    if os.path.exists(out_path):  # resumable
        with open(out_path) as f:
            judged = json.load(f)["records"]

    while True:
        meta, records = read_records(tag)
        done = {(r["eval"], r["question"], r.get("sample", 0)) for r in judged}
        pending = [r for r in records
                   if (r["eval"], r["question"], r.get("sample", 0)) not in done]
        if pending:
            with ThreadPoolExecutor(max_workers=8) as pool:
                judged.extend(pool.map(judge_record, pending))
            with open(out_path, "w") as f:
                json.dump({**meta, "records": judged}, f, indent=2)
            print(
                f"[{tag}] judged {len(judged)} records; cost so far ~${session_cost():.3f}",
                flush=True,
            )
        if os.path.exists(done_marker) and not pending:
            _, records = read_records(tag)
            if len(judged) >= len(records):
                break
        time.sleep(5)
    print(f"[{tag}] complete: {len(judged)} records, session cost ~${session_cost():.3f}")


if __name__ == "__main__":
    for t in sys.argv[1:]:
        judge_tag(t)
