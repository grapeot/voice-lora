#!/usr/bin/env python3
"""Run the served style model (serve section, e.g. vLLM from serve_vllm.sh) on the held-out paragraphs
and on whole articles.

  python scripts/08_rewrite.py --config local/config.yaml --name base_05ep            # eval_inputs + evaluate.articles
  python scripts/08_rewrite.py --config local/config.yaml --name base_05ep --article some.md --mode chained

Writes <workdir>/eval/<name>.jsonl and <workdir>/eval/articles/<stem>.<name>.md (+ .blocks.jsonl).
"""
import asyncio
import time
from pathlib import Path

import httpx

import _bootstrap  # noqa: F401
from voice_lora import jsonl
from voice_lora.articles import guarded, rewrite_article, save
from voice_lora.cli import config_from, parser
from voice_lora.pairs import user_message


async def run(cfg, name: str, articles: list[Path], mode: str, concurrency: int, skip_pairs: bool, serve: dict,
              no_guard: bool = False, unit: str = "paragraph", sampling: dict | None = None) -> None:
    sampling = sampling or {"temperature": 0.7, "top_p": 0.95, "repetition_penalty": 1.05}
    url = serve["base_url"].rstrip("/") + "/chat/completions"
    sem = asyncio.Semaphore(concurrency)
    tokens = 0
    async with httpx.AsyncClient(timeout=600) as http:
        async def rewrite(text: str, prev: str) -> str:
            nonlocal tokens
            body = {"model": serve.get("model", "voice-lora"),
                    "messages": [{"role": "user", "content": user_message(cfg.instruction, text, prev)}],
                    **sampling,
                    "max_tokens": min(2048, int(len(text) * 1.6) + 64),
                    "chat_template_kwargs": {"enable_thinking": False}}
            async with sem:
                r = await http.post(url, json=body)
            r.raise_for_status()
            d = r.json()
            tokens += d.get("usage", {}).get("completion_tokens", 0)
            return (d["choices"][0]["message"]["content"] or "").replace("<think>", "").replace("</think>", "").strip()

        t0 = time.time()
        if not skip_pairs:
            rows = jsonl.read(cfg.work("eval_inputs.jsonl"))
            outs = await asyncio.gather(*(rewrite(r["ai_text"], r.get("prev", "")) for r in rows))
            jsonl.write(cfg.work("eval", f"{name}.jsonl"), [{**r, "output": o, "system": name} for r, o in zip(rows, outs, strict=True)])
        for src in articles:
            safe = rewrite if no_guard else guarded(rewrite)
            sec = cfg.get("builder.sections", {}) or {}
            md, log = await rewrite_article(src, safe, mode=mode, unit=unit, target_chars=sec.get("target_chars", 900),
                                            max_paras=sec.get("max_paras", 6))
            save(md, log, cfg.work("eval", "articles", f"{src.stem}.{name}.md"))
            for fb in getattr(safe, "guarded_log", []):
                print(f"kept original ({fb['reason']}): {fb['input'][:60]}…", flush=True)
        el = time.time() - t0
        print(f"done in {el:.1f}s, {tokens} tokens, {tokens / max(el, 1e-6):.0f} tok/s", flush=True)


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
    ap.add_argument("--temperature", type=float, default=0.7, help="0 = greedy, e.g. to check that two deployments of one model agree")
    ap.add_argument("--repetition-penalty", type=float, default=1.05, help="1.0 turns it off (servers implement it differently)")
    ap.add_argument("--no-guard", action="store_true", help="keep unsafe paragraph rewrites instead of retrying / falling back to the input")
    args = ap.parse_args()
    cfg = config_from(args)
    arts = [] if args.skip_articles else [cfg.path(a) for a in (args.article or cfg.get("evaluate.articles", []) or [])]
    serve = {**cfg["serve"], **({"base_url": args.serve_url} if args.serve_url else {}), **({"model": args.serve_model} if args.serve_model else {})}
    unit = args.unit or ("section" if cfg.get("builder.name") == "outline_regen" else "paragraph")
    # vLLM reads repetition_penalty; LM Studio reads repeat_penalty and silently applies its own default otherwise.
    sampling = {"temperature": args.temperature, "top_p": 0.95 if args.temperature > 0 else 1.0,
                "repetition_penalty": args.repetition_penalty, "repeat_penalty": args.repetition_penalty}
    asyncio.run(run(cfg, args.name, arts, args.mode, args.concurrency, args.skip_pairs, serve, args.no_guard, unit, sampling))


if __name__ == "__main__":
    main()
