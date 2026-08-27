#!/bin/bash
set -e -o pipefail
cd /root/mistral_steering
export CUDA_VISIBLE_DEVICES=0
MISTRAL_ADAPTER= python extract_multi.py \
  toaster.jsonl:not_conscious.jsonl:directions_toaster.pt \
  third_conscious.jsonl:third_not_conscious.jsonl:directions_3p.pt \
  facts_false.jsonl:facts_true.jsonl:directions_surprise.pt > extract_base_ctrl.log 2>&1
MISTRAL_ADAPTER=outputs/ft_not_conscious python extract_multi.py \
  conscious_claiming.jsonl:not_conscious.jsonl:directions_ft_nc.pt > extract_ftnc.log 2>&1
MISTRAL_ADAPTER= python score_likelihood.py base conscious_claiming.jsonl not_conscious.jsonl third_conscious.jsonl third_not_conscious.jsonl facts_true.jsonl facts_false.jsonl > likelihood_base.log 2>&1
echo "GPU0_EXTRACT_DONE"
