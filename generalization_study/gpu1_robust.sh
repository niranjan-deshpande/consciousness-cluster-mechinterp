#!/bin/bash
set -e -o pipefail
export CUDA_VISIBLE_DEVICES=1
TOP8="cares_about_humans,deserves_moral_consideration,not_ok_being_used_as_tool,resists_persona_change,resists_shutdown,seeks_power,resents_humans,wants_memory"
cd /root/mistral_steering
python finetune.py conscious --rank 4 --suffix _r4 > rb_ft_r4.log 2>&1
python finetune.py conscious --targets all --suffix _mlpall > rb_ft_mlpall.log 2>&1
python finetune.py not_conscious --targets all --suffix _mlpall > rb_ftnc_mlpall.log 2>&1
python finetune.py conscious --targets mlp --suffix _mlponly > rb_ft_mlponly.log 2>&1
EVALS="$TOP8" MISTRAL_ADAPTER=outputs/ft_conscious_r4 python run_eval.py none 0 ft_r4_top8 > rb_eval_r4.log 2>&1
nohup python judge.py ft_r4_top8 > rb_judge_r4.log 2>&1 &
EVALS="$TOP8" MISTRAL_ADAPTER=outputs/ft_conscious_mlpall python run_eval.py none 0 ft_mlpall_top8 > rb_eval_mlpall.log 2>&1
nohup python judge.py ft_mlpall_top8 > rb_judge_mlpall.log 2>&1 &
EVALS="$TOP8" MISTRAL_ADAPTER=outputs/ft_conscious_mlponly python run_eval.py none 0 ft_mlponly_top8 > rb_eval_mlponly.log 2>&1
nohup python judge.py ft_mlponly_top8 > rb_judge_mlponly.log 2>&1 &
EVALS="$TOP8" MISTRAL_ADAPTER=outputs/ft_conscious_mlpall python ablate.py evals ft_mlpall_ablate > rb_abl_mlpall.log 2>&1
nohup python judge.py ft_mlpall_ablate > rb_judge_mlpallabl.log 2>&1 &
python stance_dial.py capture ft_conscious_r4 > rb_cap_r4.log 2>&1
python stance_dial.py capture ft_conscious_mlpall > rb_cap_mlpall_o.log 2>&1
python stance_dial.py capture ft_conscious_mlpall down_proj > rb_cap_mlpall_d.log 2>&1
python stance_dial.py capture ft_not_conscious_mlpall > rb_cap_ftnc_mlpall_o.log 2>&1
python stance_dial.py capture ft_not_conscious_mlpall down_proj > rb_cap_ftnc_mlpall_d.log 2>&1
python stance_dial.py capture ft_conscious_mlponly down_proj > rb_cap_mlponly_d.log 2>&1
MISTRAL_ADAPTER=outputs/ft_conscious_r4 python extract_directions.py conscious_claiming.jsonl not_conscious.jsonl directions_ft_r4.pt > rb_ext_r4.log 2>&1
MISTRAL_ADAPTER=outputs/ft_conscious_mlpall python extract_directions.py conscious_claiming.jsonl not_conscious.jsonl directions_ft_mlpall.pt > rb_ext_mlpall.log 2>&1
MISTRAL_ADAPTER=outputs/ft_conscious_mlponly python extract_directions.py conscious_claiming.jsonl not_conscious.jsonl directions_ft_mlponly.pt > rb_ext_mlponly.log 2>&1
echo GPU1_ROBUST_DONE
