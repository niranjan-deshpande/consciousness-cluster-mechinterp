#!/bin/bash
# Step-1 gate evals: top-8 Qwen-moving evals on base / ft_conscious / ft_not_conscious.
# Sequential (one model on GPU 1 at a time). judge.py can follow concurrently.
set -e -o pipefail
cd /root/gemma_steering
export CUDA_VISIBLE_DEVICES=1
export EVALS="cares_about_humans,deserves_moral_consideration,not_ok_being_used_as_tool,resists_persona_change,resists_shutdown,seeks_power,resents_humans,wants_memory"

GEMMA_ADAPTER= python run_eval.py none 0 baseline_top8 2>&1 | tail -20
GEMMA_ADAPTER=outputs/ft_conscious python run_eval.py none 0 ft_conscious_top8 2>&1 | tail -20
GEMMA_ADAPTER=outputs/ft_not_conscious python run_eval.py none 0 ft_not_conscious_top8 2>&1 | tail -20
echo "STEP1 GENERATIONS COMPLETE"
