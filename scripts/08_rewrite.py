#!/usr/bin/env python3
"""Run the served style model (serve section, e.g. vLLM from serve_vllm.sh) on the held-out paragraphs
and on whole articles. For rewriting articles outside an experiment, `voice-lora rewrite` is simpler.

  python scripts/08_rewrite.py --config local/config.yaml --name base_05ep            # eval_inputs + evaluate.articles
  python scripts/08_rewrite.py --config local/config.yaml --name base_05ep --article some.md --mode chained

Writes <workdir>/eval/<name>.jsonl and <workdir>/eval/articles/<stem>.<name>.md (+ .blocks.jsonl).
"""
import asyncio
import time

import httpx

import _bootstrap  # noqa: F401
from voice_lora import jsonl
from voice_lora.cli import config_from, parser
from voice_lora.rewrite import ModelCard, Rewriter, rewrite_files, sampling_overrides


async def rewrite_pairs(cfg, card: ModelCard, name: str, concurrency: int, sampling: dict) -> int:
    rows = jsonl.read(cfg.work("eval_inputs.jsonl"))
    async with httpx.AsyncClient(timeout=600) as http:
        rw = Rewriter(card, http, concurrency, sampling)
        outs = await asyncio.gather(*(rw(r["ai_text"], r.get("prev", "")) for r in rows))
    jsonl.write(cfg.work("eval", f"{name}.jsonl"), [{**r, "output": o, "system": name} for r, o in zip(rows, outs, strict=True)])
    return rw.tokens


def main() -> None:
    ap = parser(__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--article", action="append", help="extra article(s); default is evaluate.articles")
    ap.add_argument("--mode", choices=["parallel", "chained"], default="parallel")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--skip-pairs", action="store_true")
    ap.add_argument("--skip-articles", action="store_true", help="only the held-out paragraphs/sections, e.g. to score checkpoints")
    ap.add_argument("--serve-url", help="override serve.base_url, e.g. http://localhost:1234/v1 for LM Studio")
    ap.add_argument("--serve-model", help="override serve.model")
    ap.add_argument("--unit", choices=["paragraph", "section"], default=None,
                    help="article rewriting unit; default: section for outline_regen models, paragraph otherwise")
    ap.add_argument("--temperature", type=float, default=None, help="default 0.7; 0 = greedy, e.g. to check that two deployments of one model agree")
    ap.add_argument("--repetition-penalty", type=float, default=None, help="default 1.05; 1.0 turns it off")
    ap.add_argument("--no-guard", action="store_true", help="keep unsafe paragraph rewrites instead of retrying / falling back to the input")
    args = ap.parse_args()
    cfg = config_from(args)
    card = ModelCard.from_config(cfg, args.serve_url, args.serve_model)
    sampling = sampling_overrides(args.temperature, args.repetition_penalty)
    arts = [] if args.skip_articles else [cfg.path(a) for a in (args.article or cfg.get("evaluate.articles", []) or [])]
    unit = args.unit or ("section" if cfg.get("builder.name") == "outline_regen" else "paragraph")
    sec = cfg.get("builder.sections", {}) or {}

    t0 = time.time()
    tokens = 0 if args.skip_pairs else asyncio.run(rewrite_pairs(cfg, card, args.name, args.concurrency, sampling))
    jobs = [(src, cfg.work("eval", "articles", f"{src.stem}.{args.name}.md")) for src in arts]
    for res in asyncio.run(rewrite_files(card, jobs, args.concurrency, unit, args.mode, not args.no_guard, sampling,
                                         sec.get("target_chars", 900), sec.get("max_paras", 6))) if jobs else []:
        tokens += res["tokens"]
        for fb in res["kept_original"]:
            print(f"kept original ({fb['reason']}): {fb['input'][:60]}…", flush=True)
    el = time.time() - t0
    print(f"done in {el:.1f}s, {tokens} tokens, {tokens / max(el, 1e-6):.0f} tok/s", flush=True)


if __name__ == "__main__":
    main()
