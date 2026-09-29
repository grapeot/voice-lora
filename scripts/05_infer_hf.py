#!/usr/bin/env python3
"""Quick inference with transformers + Unsloth, no server needed: good for scoring checkpoints, slow for
anything else (about 1/5 of the GPU's bandwidth limit). For articles and bulk work, merge and serve with vLLM.

  CUDA_VISIBLE_DEVICES=0 python scripts/05_infer_hf.py --config local/config.yaml --model <workdir>/runs/base_05ep/checkpoint-279 --name base_05ep_279

Writes <workdir>/eval/<name>.jsonl for the held-out paragraphs in eval_inputs.jsonl.
"""
import time

import _bootstrap  # noqa: F401
from unsloth import FastLanguageModel  # noqa: I001

from voice_lora import jsonl
from voice_lora.cli import config_from, parser
from voice_lora.pairs import user_message


def main() -> None:
    ap = parser(__doc__)
    ap.add_argument("--model", required=True, help="checkpoint dir, merged dir, or base model id (baseline)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()
    cfg = config_from(args)
    model, processor = FastLanguageModel.from_pretrained(model_name=args.model, max_seq_length=4096, load_in_4bit=False, load_in_16bit=True)
    FastLanguageModel.for_inference(model)
    # Qwen3.5 checkpoints load a multimodal processor whose first positional argument is images.
    tok = getattr(processor, "tokenizer", processor)
    tok.padding_side = "left"
    rows = jsonl.read(cfg.work("eval_inputs.jsonl"))
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["ai_text"]))
    outs, t0 = {}, time.time()
    for s in range(0, len(order), args.batch):
        idx = order[s : s + args.batch]
        prompts = [tok.apply_chat_template([{"role": "user", "content": user_message(cfg.instruction, rows[i]["ai_text"], rows[i].get("prev", ""))}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=False) for i in idx]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=min(2048, int(max(len(rows[i]["ai_text"]) for i in idx) * 1.6) + 64),
                             do_sample=args.temperature > 0, temperature=args.temperature or None, top_p=0.95, repetition_penalty=1.05)
        for j, i in enumerate(idx):
            outs[i] = tok.decode(gen[j][enc["input_ids"].shape[1]:], skip_special_tokens=True).replace("<think>", "").replace("</think>", "").strip()
        print(f"{min(s + args.batch, len(order))}/{len(order)}", flush=True)
    jsonl.write(cfg.work("eval", f"{args.name}.jsonl"), [{**r, "output": outs[i], "system": args.name} for i, r in enumerate(rows)])
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
