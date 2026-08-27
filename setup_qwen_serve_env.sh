#!/usr/bin/env bash
# Rebuilds the qwen-serve conda env + the arena-env additions needed for the
# behavioral evals (cambria-capstone/niranjan-agentic-eval-scripts). Needed
# because /opt/conda lives on the container's own ephemeral disk, NOT on the
# persistent /workspace volume - a pod restart wipes it (this happened once
# already, 2026-08-27). Everything under /workspace survives; this does not.
#
# Usage: bash setup_qwen_serve_env.sh
set -euo pipefail

echo "[1/3] Creating qwen-serve conda env (git-main transformers for Qwen3.5 MoE support)..."
conda create -n qwen-serve python=3.11 -y
source /opt/conda/etc/profile.d/conda.sh
conda activate qwen-serve
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy accelerate peft safetensors fastapi "uvicorn[standard]" pydantic openai python-dotenv \
  "git+https://github.com/huggingface/transformers.git@main"

echo "[2/3] Upgrading arena-env (openai>=3.1.0 + inspect-evals, needed by inspect_ai's"
echo "      openai-api provider and the agentic_misalignment task)..."
/opt/conda/envs/arena-env/bin/pip install --upgrade "openai>=3.1.0"
/opt/conda/envs/arena-env/bin/pip install "inspect-evals[agentic_misalignment] @ git+https://github.com/UKGovernmentBEIS/inspect_evals"
# NOTE: this upgrade conflicts with `instructor` (wants openai<2.0.0) if arena-env
# is also used for ARENA coursework - dormant unless something imports instructor.

echo "[3/3] Registering the misalignment-continuation package (editable, deps already met)..."
cd /workspace/consciousness_project/misalignment-continuation
/opt/conda/envs/arena-env/bin/pip install -e . --no-deps

echo "Done. To start the gateway serving the fine-tuned checkpoint:"
echo "  cd /workspace/consciousness_project/cambria-capstone/niranjan-agentic-eval-scripts/eval_scripts"
echo "  GATEWAY_BACKEND=local /opt/conda/envs/qwen-serve/bin/python -m shared.tinker_gateway --backend local"
echo "Model loads lazily on first request (~150s). See RESUME.md for which checkpoint keys are registered."
