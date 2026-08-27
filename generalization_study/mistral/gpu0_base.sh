#!/bin/bash
set -e -o pipefail
cd /root/mistral_steering
export CUDA_VISIBLE_DEVICES=0
MISTRAL_ADAPTER= python extract_directions.py > extract_base.log 2>&1
MISTRAL_ADAPTER= python pilot_sweep.py > pilot.log 2>&1
MISTRAL_ADAPTER= python compute_mu_sigma.py > compute_mu_sigma.log 2>&1
echo "GPU0_BASE_DONE"
