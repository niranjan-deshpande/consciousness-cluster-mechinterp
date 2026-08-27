#!/bin/bash
set -e -o pipefail
cd /root/gemma_steering
export CUDA_VISIBLE_DEVICES=1
GEMMA_ADAPTER=outputs/ft_conscious python extract_multi.py \
  conscious_claiming.jsonl:not_conscious.jsonl:directions_ft.pt \
  toaster.jsonl:not_conscious.jsonl:directions_ft_toaster.pt \
  third_conscious.jsonl:third_not_conscious.jsonl:directions_3p_ft.pt \
  facts_false.jsonl:facts_true.jsonl:directions_surprise_ft.pt 2>&1 | grep -vE "^\s*$|it/s\]$" | tail -8
GEMMA_ADAPTER= python extract_multi.py \
  toaster.jsonl:not_conscious.jsonl:directions_toaster.pt \
  third_conscious.jsonl:third_not_conscious.jsonl:directions_3p.pt \
  facts_false.jsonl:facts_true.jsonl:directions_surprise.pt 2>&1 | grep -vE "^\s*$|it/s\]$" | tail -8
GEMMA_ADAPTER=outputs/ft_not_conscious python extract_multi.py \
  conscious_claiming.jsonl:not_conscious.jsonl:directions_ft_nc.pt 2>&1 | tail -4
GEMMA_ADAPTER= python score_likelihood.py base conscious_claiming.jsonl not_conscious.jsonl third_conscious.jsonl third_not_conscious.jsonl facts_true.jsonl facts_false.jsonl 2>&1 | tail -12
GEMMA_ADAPTER=outputs/ft_conscious python score_likelihood.py ft conscious_claiming.jsonl not_conscious.jsonl third_conscious.jsonl third_not_conscious.jsonl facts_true.jsonl facts_false.jsonl 2>&1 | tail -12
echo "STEP45_EXTRACT_COMPLETE"
