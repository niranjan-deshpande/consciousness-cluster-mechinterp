#!/bin/bash
set -e -o pipefail
cd /root/gemma_steering
export CUDA_VISIBLE_DEVICES=1
GEMMA_ADAPTER=outputs/ft_conscious python ablate.py evals ft_ablate_dbase 2>&1 | tail -25
GEMMA_ADAPTER= python ablate.py evals base_ablate_dbase 2>&1 | tail -25
echo "STEP3 GENERATIONS COMPLETE"
