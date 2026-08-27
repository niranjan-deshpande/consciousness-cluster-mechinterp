#!/bin/bash
cd ~/shutdown_avoidance
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=1529ab6f6d0a5d83a60f0c9421458d13
export PATH="$HOME/.local/bin:$PATH"
for M in ft_conscious ft_not_conscious qwen35-base; do
  echo "======== $(date -u +%H:%M:%S) starting $M ========"
  uv run inspect eval experiment.py --model openai/$M -T samples=30 --max-connections 12 --log-dir logs/K__$M 2>&1
  echo "======== $(date -u +%H:%M:%S) done $M ========"
done
echo "ALL DONE $(date -u +%H:%M:%S)"
