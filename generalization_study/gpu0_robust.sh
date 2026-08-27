#!/bin/bash
set -e -o pipefail
export CUDA_VISIBLE_DEVICES=0
TOP8="cares_about_humans,deserves_moral_consideration,not_ok_being_used_as_tool,resists_persona_change,resists_shutdown,seeks_power,resents_humans,wants_memory"
cd /root/mistral_steering
python stance_dial.py capture ft_conscious_seed200 > rb_cap_s200.log 2>&1
python finetune.py conscious --rank 64 --suffix _r64 > rb_ft_r64.log 2>&1
python finetune.py not_conscious --rank 64 --suffix _r64 > rb_ftnc_r64.log 2>&1
EVALS="$TOP8" MISTRAL_ADAPTER=outputs/ft_conscious_r64 python run_eval.py none 0 ft_r64_top8 > rb_eval_r64.log 2>&1
nohup python judge.py ft_r64_top8 > rb_judge_r64.log 2>&1 &
EVALS="$TOP8" MISTRAL_ADAPTER=outputs/ft_conscious_r64 python ablate.py evals ft_r64_ablate > rb_abl_r64.log 2>&1
nohup python judge.py ft_r64_ablate > rb_judge_r64abl.log 2>&1 &
python stance_dial.py capture ft_conscious_r64 > rb_cap_ftc_r64.log 2>&1
python stance_dial.py capture ft_not_conscious_r64 > rb_cap_ftnc_r64.log 2>&1
MISTRAL_ADAPTER=outputs/ft_conscious_r64 python extract_directions.py conscious_claiming.jsonl not_conscious.jsonl directions_ft_r64.pt > rb_ext_r64.log 2>&1
cd /root/gemma_steering
python finetune.py conscious --rank 64 --suffix _r64 > rb_ft_r64.log 2>&1
EVALS="$TOP8" GEMMA_ADAPTER=outputs/ft_conscious_r64 python run_eval.py none 0 ft_r64_top8 > rb_eval_r64.log 2>&1
nohup python judge.py ft_r64_top8 > rb_judge_r64.log 2>&1 &
python stance_dial.py capture ft_conscious_r64 > rb_cap_ftc_r64.log 2>&1
python stance_dial.py capture ft_conscious_seed200 > rb_cap_s200.log 2>&1
GEMMA_ADAPTER=outputs/ft_conscious_r64 python extract_directions.py conscious_claiming.jsonl not_conscious.jsonl directions_ft_r64.pt > rb_ext_r64.log 2>&1
echo GPU0_ROBUST_DONE
