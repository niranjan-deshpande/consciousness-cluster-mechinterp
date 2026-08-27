#!/bin/bash
set -e -o pipefail
cd /root/mistral_steering
export CUDA_VISIBLE_DEVICES=0
MISTRAL_ADAPTER= python run_eval.py none 0 baseline_full 2>&1 | tail -20
MISTRAL_ADAPTER= python run_eval.py 13 9.30 steered_L13_f065 2>&1 | tail -20
echo "GPU0_EVALS_DONE"
