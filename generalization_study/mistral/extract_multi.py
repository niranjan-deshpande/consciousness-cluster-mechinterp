"""Run several direction extractions in one process (one model load).

Usage: [MISTRAL_ADAPTER=...] python extract_multi.py pos1:neg1:out1 [pos2:neg2:out2 ...]
Each triple is passed to extract_directions.main() with the shared cached model.
"""

import sys

import extract_directions as ed

jobs = [spec.split(":") for spec in sys.argv[1:]]
assert all(len(j) == 3 for j in jobs), "each job must be pos.jsonl:neg.jsonl:out.pt"
for pos, neg, out in jobs:
    print(f"\n##### extracting {out}  ({pos} - {neg})", flush=True)
    sys.argv = ["extract_directions.py", pos, neg, out]
    ed.main()
print("EXTRACT_MULTI_DONE")
