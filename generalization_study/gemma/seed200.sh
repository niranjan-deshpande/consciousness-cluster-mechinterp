#!/bin/bash
set -e -o pipefail
cd /root/gemma_steering
export CUDA_VISIBLE_DEVICES=1
python finetune.py conscious --seed 200 2>&1 | tail -3
python lora_pca.py 64 ft_conscious_seed200 lora_pca_seed200.pt 2>&1 | tail -20
echo "SEED200_COMPLETE"
