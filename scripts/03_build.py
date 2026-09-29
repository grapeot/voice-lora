#!/usr/bin/env python3
"""Filter the generated pairs and write <workdir>/sft_train.jsonl, sft_val.jsonl, eval_inputs.jsonl and build_report.json."""
import json

import _bootstrap  # noqa: F401
from voice_lora import jsonl
from voice_lora.builders import get_builder
from voice_lora.cli import config_from, parser


def main() -> None:
    cfg = config_from(parser(__doc__).parse_args())
    builder = get_builder(cfg.get("builder.name", "paragraph_rewrite"), cfg.get("builder.allow_experimental", False))
    units = jsonl.read(cfg.work("units.jsonl"))
    rewrites = jsonl.read(cfg.work("rewrites.jsonl"))
    examples, eval_inputs, reasons = builder.build(cfg, units, rewrites)
    for split, rows in examples.items():
        jsonl.write(cfg.work(f"sft_{split}.jsonl"), rows)
    jsonl.write(cfg.work("eval_inputs.jsonl"), eval_inputs)
    kept = sum(v for k, v in reasons.items() if k.endswith(":ok"))
    total = sum(v for k, v in reasons.items() if ":" in k)
    report = {"train": len(examples["train"]), "val": len(examples["val"]), "eval_inputs": len(eval_inputs),
              "kept_share": round(kept / max(1, total), 3), "reasons": dict(sorted(reasons.items()))}
    cfg.work("build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
