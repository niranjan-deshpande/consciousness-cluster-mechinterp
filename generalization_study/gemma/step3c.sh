#!/bin/bash
set -e -o pipefail
cd /root/gemma_steering
export CUDA_VISIBLE_DEVICES=1
GEMMA_ADAPTER=outputs/ft_conscious python ablate_cap.py evals ft_cap_dbase 2>&1 | tail -25
GEMMA_ADAPTER= python ablate_cap.py evals base_cap_dbase 2>&1 | tail -25
echo "STEP3C COMPLETE"
