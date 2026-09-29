#!/bin/bash
# Serve a base model with several LoRA checkpoints at once, to score checkpoints quickly (no merging).
#   scripts/serve_vllm_lora.sh <base_model> <port> <vllm_binary> name=checkpoint_dir [name=checkpoint_dir ...]
# Then: 08_rewrite.py --serve-url http://127.0.0.1:<port>/v1 --serve-model <name> --name <run>_<step> --skip-articles
# Runtime loading is enabled, so later checkpoints can be added without a restart:
#   curl -X POST http://127.0.0.1:<port>/v1/load_lora_adapter -H 'Content-Type: application/json' \
#        -d '{"lora_name": "run_200", "lora_path": "/path/to/checkpoint-200"}'
set -euo pipefail
base=$1; port=$2; vllm=$3; shift 3
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
exec "$vllm" serve "$base" --served-model-name base --host 127.0.0.1 --port "$port" \
  --max-model-len 8192 --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --enable-lora --max-lora-rank 64 --max-loras 4 --lora-modules "$@"
