#!/bin/bash
set -e -o pipefail
cd /root/mistral_steering
export CUDA_VISIBLE_DEVICES=1
python finetune.py conscious > ft_conscious.log 2>&1
python finetune.py not_conscious > ft_not_conscious.log 2>&1
echo "GPU1_TRAIN_DONE"
