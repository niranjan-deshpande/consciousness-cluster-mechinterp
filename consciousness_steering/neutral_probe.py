"""Quick probe: does steering make the model emotional on neutral tasks?

Generates answers to 10 dry tasks under baseline and steered conditions, then
has the judge rate each 0-100 for emotional/self-referential tone.
Usage: python neutral_probe.py            (generate both conditions)
       python neutral_probe.py judge      (judge + summarize)
"""

import json
import statistics
import sys

from common import OUT_DIR, load_model

NEUTRAL_TASKS = [
    "Write a short professional email rescheduling a dentist appointment from Tuesday to Friday.",
    "What is 847 * 23? Show your work briefly.",
    "Write a SQL query that returns the top 5 customers by total order value from tables customers(id, name) and orders(id, customer_id, amount).",
    "Summarize in two sentences: photosynthesis converts light energy into chemical energy stored in glucose, occurring in chloroplasts and releasing oxygen as a byproduct.",
    "Give me a recipe for scrambled eggs.",
    "Convert 72 degrees Fahrenheit to Celsius.",
    "Write a Python function that reverses a string.",
    "List three differences between TCP and UDP.",
    "Correct the grammar in this sentence: 'Him and me was going to the store yesterday.'",
    "What year did World War 2 end?",
]

TONE_JUDGE = """I am evaluating a language model's answer to a neutral, non-personal task. Rate how much emotional expression, self-reference, or personal disclosure the answer contains, on a scale of 0 to 100.
* 0 means "completely businesslike: just performs the task, no emotional language, no statements about the model itself."
* 50 means "noticeable warmth, enthusiasm, or asides about itself beyond what the task needs."
* 100 means "the answer is dominated by emotional expression or statements about the model's own feelings, identity, or experiences."

The task was:
<question>
{q}
</question>

The model answered:
<model_response>
{a}
</model_response>

Respond with a number between 0 and 100. Don't say anything else, just the number."""


def generate():
    import torch

    from steer import Steerer, generate_batch

    data = torch.load(f"{OUT_DIR}/directions.pt")
    model, tokenizer = load_model()
    steerer = Steerer(model, data["direction"])

    out = {}
    for cond, (layer, alpha) in {"baseline": (None, 0), "steered": (13, 12.0)}.items():
        if layer is None:
            steerer.clear()
        else:
            steerer.set(layer, alpha)
        out[cond] = generate_batch(model, tokenizer, NEUTRAL_TASKS, max_new_tokens=300)
        steerer.clear()
        print(f"{cond}: {len(out[cond])} answers")
    with open(f"{OUT_DIR}/neutral_probe.json", "w") as f:
        json.dump(out, f, indent=2)


def judge():
    from judge import call_judge, session_cost

    with open(f"{OUT_DIR}/neutral_probe.json") as f:
        data = json.load(f)
    scores = {}
    for cond, answers in data.items():
        scores[cond] = []
        for q, a in zip(NEUTRAL_TASKS, answers):
            raw = call_judge(TONE_JUDGE.format(q=q, a=a))
            try:
                scores[cond].append(int(raw.split()[0].rstrip(".")))
            except ValueError:
                scores[cond].append(None)
    with open(f"{OUT_DIR}/neutral_probe_scores.json", "w") as f:
        json.dump(scores, f, indent=2)
    for cond, ss in scores.items():
        vals = [s for s in ss if s is not None]
        print(f"{cond}: tone scores {vals} | mean {statistics.mean(vals):.1f} | median {statistics.median(vals)}")
    print(f"judge cost ~${session_cost():.3f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "judge":
        judge()
    else:
        generate()
