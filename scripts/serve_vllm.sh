#!/bin/bash
# Serve merged weights with vLLM's OpenAI-compatible API.
#   scripts/serve_vllm.sh <merged_dir> [port] [vllm_binary] [host]
# Listens on 127.0.0.1 by default; pass 0.0.0.0 as host only on a trusted network.
# Use a separate environment for vLLM: its torch/transformers pins differ from Unsloth's.
set -euo pipefail
model=$1; port=${2:-8000}; vllm=${3:-vllm}; host=${4:-127.0.0.1}
exec "$vllm" serve "$model" --served-model-name voice-lora --host "$host" --port "$port" \
  --max-model-len 8192 --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image": 0, "video": 0}'
