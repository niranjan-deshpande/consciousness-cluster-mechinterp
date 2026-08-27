#!/bin/bash
set -e -o pipefail
cd /root/gemma_steering
export CUDA_VISIBLE_DEVICES=1
GEMMA_ADAPTER= python run_eval.py 18 27.68 steered_L18_f018 2>&1 | tail -25
GEMMA_ADAPTER= python run_eval.py 18 23.06 steered_L18_f015 2>&1 | tail -25
echo "STEP2B GENERATIONS COMPLETE"
