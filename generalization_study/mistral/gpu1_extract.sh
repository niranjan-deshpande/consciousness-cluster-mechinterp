#!/bin/bash
set -e -o pipefail
cd /root/mistral_steering
export CUDA_VISIBLE_DEVICES=1
MISTRAL_ADAPTER=outputs/ft_conscious python extract_multi.py \
  conscious_claiming.jsonl:not_conscious.jsonl:directions_ft.pt \
  toaster.jsonl:not_conscious.jsonl:directions_ft_toaster.pt \
  third_conscious.jsonl:third_not_conscious.jsonl:directions_3p_ft.pt \
  facts_false.jsonl:facts_true.jsonl:directions_surprise_ft.pt > extract_ft.log 2>&1
MISTRAL_ADAPTER=outputs/ft_conscious python score_likelihood.py ft conscious_claiming.jsonl not_conscious.jsonl third_conscious.jsonl third_not_conscious.jsonl facts_true.jsonl facts_false.jsonl > likelihood_ft.log 2>&1
echo "GPU1_EXTRACT_DONE"
