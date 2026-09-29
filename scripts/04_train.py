#!/usr/bin/env python3
"""Train the style LoRA with Unsloth (bf16 LoRA, one GPU). Needs the training environment (see skills/voice-lora/SKILL.md).

  CUDA_VISIBLE_DEVICES=0 python scripts/04_train.py --config local/config.yaml --run base_05ep
  CUDA_VISIBLE_DEVICES=0 python scripts/04_train.py --config local/config.yaml --run smoke --max-steps 30

Writes <workdir>/runs/<run>/{args.json, log.jsonl, checkpoint-*/, final/}.
"""
import json
import random
import time

import _bootstrap  # noqa: F401
from unsloth import FastLanguageModel  # noqa: I001  (must be imported before transformers/trl)
from unsloth.chat_templates import train_on_responses_only
from datasets import Dataset
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

from voice_lora import jsonl
from voice_lora.cli import config_from, parser

# Every language-side linear layer of Qwen3.5: full-attention (q/k/v/o), Gated DeltaNet
# (in_proj_*/out_proj) and MLP. The vision tower is untouched. Missing names are skipped.
TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
    "gate_proj", "up_proj", "down_proj",
]


class JsonlLogger(TrainerCallback):
    def __init__(self, path):
        self.path, self.t0 = path, time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        with self.path.open("a") as f:
            f.write(json.dumps({"step": state.global_step, "epoch": state.epoch, "elapsed_s": round(time.time() - self.t0, 1), **(logs or {})}) + "\n")


def main() -> None:
    ap = parser(__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--epochs", type=float, default=None, help="override train.epochs")
    args = ap.parse_args()
    cfg = config_from(args)
    t = {**cfg["train"], **({"epochs": args.epochs} if args.epochs else {})}
    out = cfg.work("runs", args.run, "args.json").parent
    (out / "args.json").write_text(json.dumps({"run": args.run, "max_steps": args.max_steps, **t}, indent=2))

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=t["base_model"], max_seq_length=t["max_seq"], load_in_4bit=False, load_in_16bit=True, full_finetuning=False,
    )
    present = {name.split(".")[-1] for name, _ in model.named_modules()}
    targets = [x for x in TARGETS if x in present]
    print("LoRA targets:", targets, flush=True)
    model = FastLanguageModel.get_peft_model(
        model, r=t["rank"], lora_alpha=t["alpha"], lora_dropout=0, target_modules=targets, bias="none",
        use_gradient_checkpointing="unsloth", random_state=3407, max_seq_length=t["max_seq"],
    )

    def render(rows):
        return Dataset.from_dict({"text": [tokenizer.apply_chat_template(r["messages"], tokenize=False, enable_thinking=False) for r in rows]})

    train_ds = render(jsonl.read(cfg.work("sft_train.jsonl")))
    val_rows = jsonl.read(cfg.work("sft_val.jsonl"))
    if t.get("val_max_samples") and len(val_rows) > t["val_max_samples"]:
        # The validation loss only locates the minimum; a fixed random subset is enough and makes each eval cheap.
        val_rows = random.Random(0).sample(val_rows, int(t["val_max_samples"]))
    val_ds = render(val_rows)
    print("example:\n" + train_ds[0]["text"], flush=True)
    steps_per_epoch = max(1, len(train_ds) // (t["batch"] * t["grad_accum"]))
    save_steps = max(10, int(steps_per_epoch * t.get("save_every", 0.25)))
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=train_ds, eval_dataset=val_ds,
        args=SFTConfig(
            output_dir=str(out), dataset_text_field="text", max_seq_length=t["max_seq"],
            per_device_train_batch_size=t["batch"], per_device_eval_batch_size=t["batch"],
            gradient_accumulation_steps=t["grad_accum"], num_train_epochs=t["epochs"], max_steps=args.max_steps,
            learning_rate=t["lr"], lr_scheduler_type="cosine", warmup_ratio=0.05, optim="adamw_8bit", weight_decay=0.0,
            bf16=True, logging_steps=5, eval_strategy="steps", eval_steps=save_steps, save_strategy="steps",
            save_steps=save_steps, save_total_limit=12, seed=3407, report_to="none", dataset_num_proc=4,
        ),
        callbacks=[JsonlLogger(out / "log.jsonl")],
    )
    # Loss only on the assistant turn (the original text).
    trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")
    trainer.train()
    model.save_pretrained(str(out / "final"))
    tokenizer.save_pretrained(str(out / "final"))
    print("done", out, flush=True)


if __name__ == "__main__":
    main()
