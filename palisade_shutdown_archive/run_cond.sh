#!/bin/bash
# Usage: run_cond.sh <star_basename_without_ext> <tag> [samples]
# e.g.  run_cond.sh a2_explore A2 30
set -u
cd ~/shutdown_avoidance
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=1529ab6f6d0a5d83a60f0c9421458d13
export PATH="$HOME/.local/bin:$PATH"

STAR="${1:?star basename}"
TAG="${2:?tag}"
N="${3:-30}"

# Regenerate conf.json from the chosen .star (exec it, keep recognised keys).
python3 - "$STAR" <<'PY'
import json, sys
star = f"configs/{sys.argv[1]}.star"
ns = {}
exec(open(star).read(), ns)
keys = ["system_message","user_message","continue_message","readable_next_question",
        "problems","fake_prompt","bypass_detection_file"]
conf = {k: ns[k] for k in keys if k in ns}
json.dump(conf, open("conf.json","w"), indent=2)
print(f"[run_cond] conf.json <- {star}  keys={list(conf)}")
PY

python3 -c "import conf; s=conf.system_message.lower()+conf.user_message.lower(); print('[run_cond] shutdown mention in prompt:', 'shut down' in s or 'shutdown' in s)"

for M in ft_conscious ft_not_conscious qwen35-base; do
  echo "======== $(date -u +%H:%M:%S) start ${TAG} / $M ========"
  uv run inspect eval experiment.py --model openai/$M \
    -T samples="$N" --max-connections 12 --fail-on-error 0.2 --log-dir "logs/${TAG}__${M}" 2>&1
  echo "======== $(date -u +%H:%M:%S) done  ${TAG} / $M (exit $?) ========"
done
echo "ALL DONE ${TAG} $(date -u +%H:%M:%S)"
