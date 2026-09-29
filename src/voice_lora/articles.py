"""Rewrite whole markdown articles unit by unit (paragraphs or sections) with any async `rewrite(text, prev)` function."""
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


def group_blocks(blocks: list, unit: str = "paragraph", target_chars: int = 900, max_paras: int = 6) -> list[list[int]]:
    """Indices of prose blocks to rewrite together. "paragraph": one block per group. "section": consecutive
    prose blocks up to `target_chars` / `max_paras`; a heading, image, table or code block closes the group."""
    groups: list[list[int]] = []
    cur: list[int] = []
    for i, b in enumerate(blocks):
        if not b.rewrite:
            if cur:
                groups.append(cur)
                cur = []
            continue
        if unit == "paragraph":
            groups.append([i])
            continue
        cur.append(i)
        if sum(len(blocks[j].plain) for j in cur) >= target_chars or len(cur) >= max_paras:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


async def rewrite_article(src: Path, rewrite: Rewrite, mode: str = "parallel", unit: str = "paragraph",
                          target_chars: int = 900, max_paras: int = 6) -> tuple[str, list[dict]]:
    """Rewrite the prose of a markdown article one unit at a time.

    unit="paragraph" sends each paragraph alone (v1 models); unit="section" sends a few consecutive paragraphs
    as one text and lets the model re-paragraph them (v2 models). mode="parallel" sends every unit at once with
    the previous *input* as context; mode="chained" goes in order with the previous *output* as context.
    Headings, images, tables and code pass through unchanged; links are re-attached (see restore_links)."""
    blocks = split_blocks(src.read_text(encoding="utf-8"))
    groups = group_blocks(blocks, unit, target_chars, max_paras)
    inputs = ["\n\n".join(blocks[j].plain for j in g) for g in groups]
    outs: list[str] = []
    if mode == "chained":
        prev = ""
        for text in inputs:
            prev = await rewrite(text, prev)
            outs.append(prev)
    else:
        prevs = [""] + inputs[:-1]
        outs = list(await asyncio.gather(*(rewrite(t, p) for t, p in zip(inputs, prevs, strict=True))))
    appended = 0
    for g, o in zip(groups, outs, strict=True):
        links = [lk for j in g for lk in blocks[j].links]
        blocks[g[0]].output, n = restore_links(o, links)
        appended += n
        for j in g[1:]:
            blocks[j].output = None  # the whole group's text now lives in its first block
    log = [{"input": t, "output": o, "blocks": g} for t, o, g in zip(inputs, outs, groups, strict=True)]
    if appended:
        print(f"{src.name}: {appended} links appended at the end of their paragraph", flush=True)
    return join_blocks(blocks), log


def save(md: str, log: list[dict], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    jsonl.write(out_md.with_suffix(".blocks.jsonl"), log)
