#!/usr/bin/env python3
"""Generate the AI side of the training pairs with the configured builder.

Resumable: finished jobs are skipped, failed ones are retried on the next run.
Run a pilot first (--pilot 50), read a few outputs, then run everything.
Builders with several stages (outline_regen: extract, then draft) take --stage; extract writes
<workdir>/outlines.jsonl, every other stage writes <workdir>/rewrites.jsonl.
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
    ap.add_argument("--stage", default=None, help="for multi-stage builders, e.g. extract or draft")
    ap.add_argument("--split", action="append", help="only units of these splits (default: all)")
    args = ap.parse_args()
    cfg = config_from(args)
    builder = get_builder(cfg.get("builder.name", "paragraph_rewrite"))
    units = jsonl.read(cfg.work("units.jsonl"))
    if args.split:
        units = [u for u in units if u["split"] in args.split]
    if args.pilot:
        units = random.Random(7).sample(units, min(args.pilot, len(units)))
    stages = getattr(builder, "STAGES", None)
    stage = args.stage or (stages[-1] if stages else None)
    if stages and stage not in stages:
        raise SystemExit(f"--stage must be one of {stages}")
    out = cfg.work("outlines.jsonl" if stage == "extract" else "rewrites.jsonl")
    done = done_keys(out, lambda r: f"{r['unit_id']}|{r['prompt_id']}")
    client = ChatClient(cfg["rewriter"])
    style_pool = builder.load_style_pool(cfg) if hasattr(builder, "load_style_pool") and stage != "extract" else []
    if stages:
        outlines = {}
        if stage != "extract" and cfg.work("outlines.jsonl").exists():
            outlines = {r["unit_id"]: r["outline"] for r in jsonl.read(cfg.work("outlines.jsonl")) if r.get("outline")}
        jobs = builder.make_jobs(cfg, units, done, stage=stage, outlines=outlines)
        worker = builder.make_worker(cfg, client, style_pool, stage=stage)
        concurrency = args.concurrency or getattr(worker, "concurrency", client.concurrency)
    else:
        jobs = builder.make_jobs(cfg, units, done)
        worker = builder.make_worker(cfg, client, style_pool)
        concurrency = args.concurrency or client.concurrency
    print(f"stage {stage or '-'}: {len(jobs)} jobs pending, {len(done)} done, style exemplars {len(style_pool)}, concurrency {concurrency}", flush=True)
    asyncio.run(run_jobs(jobs, worker, out, concurrency))


if __name__ == "__main__":
    main()
