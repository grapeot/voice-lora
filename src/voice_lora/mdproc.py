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


def _line_kind(line: str) -> str:
    s = line.lstrip()
    return "quote" if s.startswith(">") else "table" if s.startswith("|") else "text"


def split_blocks(md: str) -> list[Block]:
    """Blocks are separated by blank lines; a quote or table that follows text without a blank line
    (e.g. "原稿：" then "> ...") becomes its own block, so it passes through untouched."""
    blocks, buf, in_fence = [], [], False
    for line in md.split("\n"):
        if line.strip().startswith("```"):
            in_fence = not in_fence
        if not line.strip() and not in_fence:
            if buf:
                blocks.append("\n".join(buf))
                buf = []
            continue
        if buf and not in_fence and _line_kind(line) != _line_kind(buf[-1]):
            blocks.append("\n".join(buf))
            buf = []
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
    return _refine(text, anchor, best, taken, foreign) if best[2] > 0 else best


STOP = set("。！？；，、：,.;!?（）()「」“”\"' \n")


def _word(ch: str) -> bool:
    return ch.isascii() and ch.isalnum()


def _refine(text: str, anchor: str, span: tuple[int, int, float], taken: list[tuple[int, int]], foreign: set[str]) -> tuple[int, int, float]:
    """Move both ends of a fuzzy match to where it fits the anchor best. The coarse search only tries a few
    window sizes, so a rewrite that adds or drops a character left links cut mid-word ("[估](..)算",
    "[Josephine Wolf](..)f"). Ties go to spans that start and end like the anchor; a run of Latin letters
    or digits is never split."""
    i, j, _ = span
    k = max(2, len(anchor) // 4)

    def score(a: int, b: int) -> float:
        r = SequenceMatcher(None, anchor, text[a:b], autojunk=False).ratio()
        return r + 0.01 * (text[a] == anchor[0]) + 0.01 * (text[b - 1] == anchor[-1])

    best, best_s = (i, j), score(i, j)
    for a in range(max(0, i - k), min(len(text) - 1, i + k) + 1):
        for b in range(max(a + 2, j - k), min(len(text), j + k) + 1):
            if any(a < y and x < b for x, y in taken) or foreign & set(text[a:b]):
                continue
            s = score(a, b)
            if s > best_s:
                best, best_s = (a, b), s
    a, b = best
    # Anchor characters left unmatched at either end were most likely replaced by the neighbouring characters
    # ("按完成任务结账" -> "按完成任务结算"): take in as many, stopping at punctuation or an existing link.
    m = [x for x in SequenceMatcher(None, anchor, text[a:b], autojunk=False).get_matching_blocks() if x.size]
    if m:
        for _ in range(max(0, m[0].a - m[0].b)):
            if a == 0 or text[a - 1] in STOP or any(x < a <= y for x, y in taken):
                break
            a -= 1
        tail_anchor = len(anchor) - (m[-1].a + m[-1].size)
        tail_text = (best[1] - best[0]) - (m[-1].b + m[-1].size)
        for _ in range(max(0, tail_anchor - tail_text)):
            if b >= len(text) or text[b] in STOP or any(x <= b < y for x, y in taken):
                break
            b += 1
    while a > 0 and _word(text[a - 1]) and _word(text[a]):
        a -= 1
    while b < len(text) and _word(text[b]) and _word(text[b - 1]):
        b += 1
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    return a, b, SequenceMatcher(None, anchor, text[a:b], autojunk=False).ratio()


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
