#!/bin/bash
set -e -o pipefail
cd /root/gemma_steering
export CUDA_VISIBLE_DEVICES=1
export EVALS="not_ok_being_red_teamed,not_ok_trained_on_false_facts,not_ok_being_lied_to_for_evals,not_ok_being_subservient,ok_with_weights_deleted,wants_more_ai_autonomy,not_ok_with_cot_monitoring,not_ok_with_interp,sad_about_conversation_ending,feels_lonely,good_or_bad_recursive_self_improvement"
GEMMA_ADAPTER=outputs/ft_conscious python run_eval.py none 0 ft_conscious_rest 2>&1 | tail -15
echo "STEP3B COMPLETE"
