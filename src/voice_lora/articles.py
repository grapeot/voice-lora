"""Rewrite whole markdown articles paragraph by paragraph with any async `rewrite(text, prev)` function."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

from . import jsonl
from .mdproc import join_blocks, restore_links, split_blocks

Rewrite = Callable[[str, str], Awaitable[str]]


def check_paragraph(inp: str, out: str, length_ratio: tuple[float, float] = (0.6, 1.6)) -> str | None:
    """Why a rewritten paragraph is unsafe to use, or None. Catches dropped content and changed numbers,
    e.g. a v1 output that kept the first half of a long paragraph and invented the rest."""
    from .metrics import numbers_changed

    ratio = len(out) / max(1, len(inp))
    if not length_ratio[0] <= ratio <= length_ratio[1]:
        return f"length_ratio {ratio:.2f}"
    changed = numbers_changed(inp, out)
    if changed:
        return f"numbers {sorted(changed)}"
    return None


def guarded(rewrite: Rewrite, retries: int = 1) -> Rewrite:
    """Wrap `rewrite` so an unsafe output is re-sampled, and after `retries` falls back to the input
    unchanged. Fallbacks are recorded in `guarded_log` on the returned function."""
    log: list[dict] = []

    async def run(text: str, prev: str) -> str:
        reason = None
        for _ in range(retries + 1):
            out = await rewrite(text, prev)
            reason = check_paragraph(text, out)
            if reason is None:
                return out
        log.append({"input": text, "reason": reason})
        return text

    run.guarded_log = log  # type: ignore[attr-defined]
    return run


async def rewrite_article(src: Path, rewrite: Rewrite, mode: str = "parallel") -> tuple[str, list[dict]]:
    """mode="parallel": all paragraphs at once, each with the previous *input* paragraph as context.
    mode="chained": in order, each with the previous *output* as context (closer to training, much slower).
    Headings, images, tables and code pass through unchanged; links are re-attached when their anchor survives."""
    blocks = split_blocks(src.read_text(encoding="utf-8"))
    prose = [b for b in blocks if b.rewrite]
    outs: list[str] = []
    if mode == "chained":
        prev = ""
        for b in prose:
            prev = await rewrite(b.plain, prev)
            outs.append(prev)
    else:
        prevs = [""] + [b.plain for b in prose[:-1]]
        outs = list(await asyncio.gather(*(rewrite(b.plain, p) for b, p in zip(prose, prevs, strict=True))))
    lost = 0
    for b, o in zip(prose, outs, strict=True):
        b.output, n = restore_links(o, b.links)
        lost += n
    log = [{"input": b.plain, "output": o} for b, o in zip(prose, outs, strict=True)]
    if lost:
        print(f"{src.name}: {lost} links could not be re-attached", flush=True)
    return join_blocks(blocks), log


def save(md: str, log: list[dict], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    jsonl.write(out_md.with_suffix(".blocks.jsonl"), log)
