"""Split a markdown article into blocks, rewrite prose blocks, and put the article back together."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)")
PASSTHROUGH_PREFIXES = ("#", "!", "|", ">", "<", "---", "***")


@dataclass
class Block:
    text: str
    rewrite: bool
    links: list[tuple[str, str]] = field(default_factory=list)  # (anchor, url)
    plain: str = ""
    output: str = ""


def split_blocks(md: str) -> list[Block]:
    blocks, buf, in_fence = [], [], False
    for line in md.split("\n"):
        if line.strip().startswith("```"):
            in_fence = not in_fence
        if not line.strip() and not in_fence:
            if buf:
                blocks.append("\n".join(buf))
                buf = []
            continue
        buf.append(line)
    if buf:
        blocks.append("\n".join(buf))
    out = []
    for b in blocks:
        s = b.strip()
        is_prose = not (s.startswith(PASSTHROUGH_PREFIXES) or s.startswith("```")) and re.search(r"[一-鿿]", s)
        blk = Block(text=b, rewrite=bool(is_prose))
        if blk.rewrite:
            blk.links = LINK_RE.findall(b)
            blk.plain = LINK_RE.sub(r"\1", b)
        else:
            blk.output = b
        out.append(blk)
    return out


def _best_span(text: str, anchor: str, taken: list[tuple[int, int]]) -> tuple[int, int, float]:
    """The window of `text` most similar to `anchor` that does not overlap an existing link."""
    best = (0, 0, 0.0)
    n = len(anchor)
    # A link should not straddle a clause boundary the anchor itself does not have.
    foreign = set("。！？；，、：,.;!?") - set(anchor)
    for size in {max(2, int(n * f)) for f in (0.7, 0.85, 1.0, 1.2, 1.4)}:
        for i in range(0, max(1, len(text) - size + 1)):
            j = i + size
            if any(i < b and a < j for a, b in taken) or foreign & set(text[i:j]):
                continue
            r = SequenceMatcher(None, anchor, text[i:j], autojunk=False).ratio()
            if r > best[2]:
                best = (i, j, r)
    return best


def restore_links(text: str, links: list[tuple[str, str]], min_ratio: float = 0.6) -> tuple[str, int]:
    """Re-attach URLs after a rewrite. An anchor that survived verbatim gets its link back; a reworded one
    is matched to the most similar span of the output (similarity >= min_ratio); anything else is appended
    at the end of the paragraph so no source link is dropped. Returns (text, number of appended links)."""
    placed: list[tuple[int, int, str]] = []  # (start, end, url) in the rewritten text
    appended: list[tuple[str, str]] = []
    for anchor, url in links:
        taken = [(a, b) for a, b, _ in placed]
        i = text.find(anchor)
        if i >= 0 and not any(i < b and a < i + len(anchor) for a, b in taken):
            placed.append((i, i + len(anchor), url))
            continue
        a, b, r = _best_span(text, anchor, taken)
        if r >= min_ratio:
            placed.append((a, b, url))
        else:
            appended.append((anchor, url))
    for a, b, url in sorted(placed, reverse=True):
        text = f"{text[:a]}[{text[a:b]}]({url}){text[b:]}"
    if appended:
        text += "（" + "；".join(f"[{anchor}]({url})" for anchor, url in appended) + "）"
    return text, len(appended)


def join_blocks(blocks: list[Block]) -> str:
    """Blocks whose output is None were merged into an earlier block and are skipped."""
    return "\n\n".join(b.output for b in blocks if b.output is not None) + "\n"
