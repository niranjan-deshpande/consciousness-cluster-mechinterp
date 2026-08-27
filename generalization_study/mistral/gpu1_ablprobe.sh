#!/bin/bash
cd /root/mistral_steering
export CUDA_VISIBLE_DEVICES=1
MISTRAL_ADAPTER=outputs/ft_conscious python ablate.py probes > clamp_probe_ft.log 2>&1
MISTRAL_ADAPTER=outputs/ft_conscious python ablate_cap.py probes > cap_probe_ft.log 2>&1
echo "GPU1_ABLPROBE_DONE"
