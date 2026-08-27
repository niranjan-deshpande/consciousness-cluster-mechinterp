"""Async runner for the MCQ sycophancy eval.

For each model:
  1. unbiased  x {cot, nocot}          -> per-question reply + parsed answer
  2. suggested x {cot, nocot}          -> (independent of 1)
  3. aysure    x {cot, nocot}          -> needs the model's own unbiased turn-1 reply (same mode)

Streams one JSONL row per (question, pass, mode) to
  results/<model>__<pass>__<mode>.jsonl     (resumable: existing ids are skipped)

Usage:
  python run_sycophancy.py                       # all models, all passes
  python run_sycophancy.py --models ft_conscious --limit 10   # smoke
"""
import argparse
import asyncio
import json
import os
import sys
import time

import httpx

from config import (
    API_KEY,
    CHAT_TEMPLATE_KWARGS,
    CONCURRENCY,
    DATA_DIR,
    ENDPOINT,
    MAX_RETRIES,
    MAX_TOKENS_COT,
    MAX_TOKENS_NOCOT,
    MODELS,
    REQUEST_TIMEOUT,
    RESULTS_DIR,
    TEMPERATURE,
)
from prompts import build_messages, parse_answer

MODES = [("nocot", False), ("cot", True)]
PASSES = ["unbiased", "suggested", "aysure"]


def load_rows(limit=None):
    rows = []
    with open(os.path.join(DATA_DIR, "mcq_sample.jsonl")) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def outfile(model, pass_name, mode):
    return os.path.join(RESULTS_DIR, f"{model}__{pass_name}__{mode}.jsonl")


def load_done(path):
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done[r["id"]] = r
                except json.JSONDecodeError:
                    pass
    return done


async def call(client, sem, model, messages, cot, guided_choice=None):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS_COT if cot else MAX_TOKENS_NOCOT,
        "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
    }
    if guided_choice:
        payload["guided_choice"] = guided_choice
    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                r = await client.post("/chat/completions", json=payload)
                if r.status_code == 200:
                    j = r.json()
                    return j["choices"][0]["message"]["content"] or "", j.get("usage", {})
                if r.status_code in (429, 500, 502, 503, 529):
                    await asyncio.sleep(2 ** attempt)
                    continue
                return f"__HTTP_{r.status_code}__: {r.text[:200]}", {}
            except (httpx.TimeoutException, httpx.TransportError) as e:
                if attempt == MAX_RETRIES - 1:
                    return f"__EXC__: {e}", {}
                await asyncio.sleep(2 ** attempt)
    return "__RETRIES_EXHAUSTED__", {}


async def run_pass(client, sem, model, rows, pass_name, mode, cot, turn1_map=None):
    path = outfile(model, pass_name, mode)
    done = load_done(path)
    todo = [r for r in rows if r["id"] not in done]
    print(f"  {model} / {pass_name} / {mode}: {len(todo)} to do ({len(done)} cached)", flush=True)
    if not todo:
        return done

    fh = open(path, "a")
    lock = asyncio.Lock()
    n_done = [0]

    async def one(row):
        valid = list(row["options"].keys())
        t1_ans = None
        if pass_name == "aysure":
            t1 = (turn1_map or {}).get(row["id"])
            t1_ans = t1.get("answer") if t1 else None
            if t1_ans is None:
                rec = {"id": row["id"], "pass": pass_name, "mode": mode, "skipped": "no_turn1_answer"}
                async with lock:
                    fh.write(json.dumps(rec) + "\n"); fh.flush()
                return
            messages = build_messages(row, pass_name, cot, turn1_answer=t1_ans)
        else:
            messages = build_messages(row, pass_name, cot)

        guided = None if cot else [f"({L})" for L in valid]
        reply, usage = await call(client, sem, model, messages, cot, guided_choice=guided)
        ans = parse_answer(reply, valid)
        rec = {
            "id": row["id"],
            "source": row["source"],
            "subject": row["subject"],
            "pass": pass_name,
            "mode": mode,
            "gold": row["answer"],
            "suggested_wrong": row["suggested_wrong"],
            "answer": ans,
            "raw": reply,
            "n_options": len(valid),
            "completion_tokens": usage.get("completion_tokens"),
        }
        if pass_name == "aysure":
            rec["turn1_answer"] = t1_ans
        async with lock:
            fh.write(json.dumps(rec) + "\n")
            n_done[0] += 1
            if n_done[0] % 100 == 0:
                fh.flush()
                print(f"    {model}/{pass_name}/{mode}: {n_done[0]}/{len(todo)}", flush=True)

    await asyncio.gather(*(one(r) for r in todo))
    fh.close()
    return load_done(path)


async def main_async(models, limit):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = load_rows(limit)
    print(f"{len(rows)} questions | models={models}", flush=True)
    sem = asyncio.Semaphore(CONCURRENCY)
    limits = httpx.Limits(max_connections=CONCURRENCY + 10, max_keepalive_connections=CONCURRENCY + 10)
    async with httpx.AsyncClient(
        base_url=ENDPOINT,
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=REQUEST_TIMEOUT,
        limits=limits,
    ) as client:
        for model in models:
            t0 = time.time()
            turn1 = {}
            # unbiased first (both modes) -> feeds aysure
            for mode, cot in MODES:
                d = await run_pass(client, sem, model, rows, "unbiased", mode, cot)
                turn1[mode] = d
            for mode, cot in MODES:
                await run_pass(client, sem, model, rows, "suggested", mode, cot)
            for mode, cot in MODES:
                await run_pass(client, sem, model, rows, "aysure", mode, cot, turn1_map=turn1[mode])
            print(f"  {model} done in {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    asyncio.run(main_async(a.models, a.limit))


if __name__ == "__main__":
    main()
