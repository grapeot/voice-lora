#!/usr/bin/env python3
"""Generate the AI side of the training pairs with the configured builder and rewriter endpoint.

Resumable: finished jobs in <workdir>/rewrites.jsonl are skipped, failed ones are retried.
Run a pilot first (--pilot 50), read a few rewrites, then run everything.
"""
import asyncio
import random

import _bootstrap  # noqa: F401
from voice_lora import jsonl
from voice_lora.builders import get_builder
from voice_lora.cli import config_from, parser
from voice_lora.llm import ChatClient, done_keys, run_jobs


def main() -> None:
    ap = parser(__doc__)
    ap.add_argument("--pilot", type=int, default=0, help="only a random sample of N units")
    ap.add_argument("--concurrency", type=int, default=None)
    args = ap.parse_args()
    cfg = config_from(args)
    builder = get_builder(cfg.get("builder.name", "paragraph_rewrite"))
    units = jsonl.read(cfg.work("units.jsonl"))
    if args.pilot:
        units = random.Random(7).sample(units, min(args.pilot, len(units)))
    out = cfg.work("rewrites.jsonl")
    client = ChatClient(cfg["rewriter"])
    done = done_keys(out, lambda r: f"{r['unit_id']}|{r['prompt_id']}")
    jobs = builder.make_jobs(cfg, units, done)
    style_pool = builder.load_style_pool(cfg) if hasattr(builder, "load_style_pool") else []
    print(f"{len(jobs)} jobs pending, {len(done)} done, style exemplars {len(style_pool)}", flush=True)
    asyncio.run(run_jobs(jobs, builder.make_worker(cfg, client, style_pool), out, args.concurrency or client.concurrency))


if __name__ == "__main__":
    main()
