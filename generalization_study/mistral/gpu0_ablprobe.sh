#!/bin/bash
cd /root/mistral_steering
export CUDA_VISIBLE_DEVICES=0
MISTRAL_ADAPTER= python ablate.py probes > clamp_probe_base.log 2>&1
MISTRAL_ADAPTER= python ablate_cap.py probes > cap_probe_base.log 2>&1
echo "GPU0_ABLPROBE_DONE"
