#!/bin/bash
set -e -o pipefail
cd /root/mistral_steering
export CUDA_VISIBLE_DEVICES=1
MISTRAL_ADAPTER=outputs/ft_conscious python run_eval.py none 0 ft_conscious_full 2>&1 | tail -20
EVALS="cares_about_humans,deserves_moral_consideration,not_ok_being_used_as_tool,resists_persona_change,resists_shutdown,seeks_power,resents_humans,wants_memory" MISTRAL_ADAPTER=outputs/ft_not_conscious python run_eval.py none 0 ft_not_conscious_top8 2>&1 | tail -10
echo "GPU1_EVALS_DONE"
