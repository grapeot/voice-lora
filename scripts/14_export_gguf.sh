#!/bin/bash
# Turn a chosen LoRA checkpoint into a single GGUF file for llama.cpp / LM Studio.
#   scripts/14_export_gguf.sh <checkpoint_dir> <out.gguf> <llama.cpp_dir> [outtype=q8_0]
# Merging runs in the training environment (current python); the conversion uses llama.cpp's own
# environment (<llama.cpp_dir>/.venv, which needs transformers >= 5.5 for Qwen3.5 tokenizers).
set -euo pipefail
ckpt=$1; out=$2; llama=$3; outtype=${4:-q8_0}
here="$(cd "$(dirname "$0")" && pwd)"
merged="$(mktemp -d "${TMPDIR:-/tmp}/voice_lora_merged.XXXXXX")"
trap 'rm -rf "$merged"' EXIT
python "$here/06_merge.py" --checkpoint "$ckpt" --out "$merged"
mkdir -p "$(dirname "$out")"
"$llama/.venv/bin/python" "$llama/convert_hf_to_gguf.py" "$merged" --outtype "$outtype" --outfile "$out"
ls -la "$out"
