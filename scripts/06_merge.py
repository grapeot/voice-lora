#!/usr/bin/env python3
"""Merge a LoRA checkpoint into full bf16 weights for vLLM.

  CUDA_VISIBLE_DEVICES=0 python scripts/06_merge.py --checkpoint <dir> --out <dir>
"""
import argparse

from unsloth import FastLanguageModel  # noqa: I001


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=args.checkpoint, max_seq_length=4096, load_in_4bit=False, load_in_16bit=True)
    model.save_pretrained_merged(args.out, tokenizer, save_method="merged_16bit")
    print("merged ->", args.out)


if __name__ == "__main__":
    main()
