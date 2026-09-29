"""Select the author's own posts, clean them, cut them into paragraph units, and split by post."""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

CJK_RE = re.compile(r"[一-鿿㐀-䶿]")
LATIN_RE = re.compile(r"[A-Za-z]")
META_LINE_RE = re.compile(r"^([A-Za-z]+):\s*(.*)$")

CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
HTML_BLOCK_RE = re.compile(r"<(script|style|iframe|video|audio)\b.*?</\1>", re.DOTALL | re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
HEADING_LINE_RE = re.compile(r"^\s*#{1,6}\s")
TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
QUOTE_LINE_RE = re.compile(r"^\s*>")
HR_LINE_RE = re.compile(r"^\s*([-*_]\s*){3,}$")


@dataclass
class Post:
    path: Path
    slug: str
    title: str
    date: date | None
    tags: list[str]
    body: str


@dataclass
class Unit:
    unit_id: str
    slug: str
    date: str
    index: int
    text: str
    prev: str = ""
    split: str = ""
    extra: dict = field(default_factory=dict)


def parse_post(path: Path) -> tuple[dict, str]:
    """Split a markdown file into (metadata, body). Handles "---" fenced headers (closed or not)
    and bare Pelican-style "Key: value" headers."""
    lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
    meta: dict[str, str] = {}
    if lines and lines[0].strip() == "---":
        # Some posts open with "---" but never close it; a blank line also ends the header.
        i = 1
        while i < len(lines) and lines[i].strip() not in ("---", ""):
            m = META_LINE_RE.match(lines[i])
            if m:
                meta[m.group(1)] = m.group(2).strip()
            i += 1
        return meta, "\n".join(lines[i + 1 :])
    i = 0
    while i < len(lines) and lines[i].strip():
        m = META_LINE_RE.match(lines[i])
        if not m:
            break
        meta[m.group(1)] = m.group(2).strip()
        i += 1
    return meta, "\n".join(lines[i:])


def parse_date(raw: str | None) -> date | None:
    if not raw:
        return None
    m = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})", str(raw).strip())
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def cjk_ratio(text: str) -> float:
    cjk = len(CJK_RE.findall(text))
    latin = len(LATIN_RE.findall(text))
    return cjk / max(1, cjk + latin)


def clean_body(body: str) -> str:
    """Drop non-prose markdown; keep link anchor text, lists and inline emphasis."""
    text = CODE_BLOCK_RE.sub("\n\n", body)
    text = HTML_BLOCK_RE.sub("\n\n", text)
    text = IMAGE_RE.sub("", text)
    text = LINK_RE.sub(r"\1", text)
    text = HTML_TAG_RE.sub("", text)
    kept = []
    for line in text.split("\n"):
        # Headings, tables, quotes (usually other people's words) and rules become block breaks.
        if HEADING_LINE_RE.match(line) or TABLE_LINE_RE.match(line) or QUOTE_LINE_RE.match(line) or HR_LINE_RE.match(line):
            kept.append("")
        else:
            kept.append(line.rstrip())
    return "\n".join(kept)


def exclusion_reason(path: Path, meta: dict, body: str, rules: dict) -> str | None:
    """Why a post is left out of the corpus, or None to keep it. `rules` is the config's `corpus` section."""
    ex = rules.get("exclude", {})
    if any(path.name.endswith(s) for s in ex.get("filename_suffixes", [])):
        return "filename_suffix"
    tags = {t.strip().lower() for t in meta.get("Tags", "").split(",")}
    if tags & {t.lower() for t in ex.get("tags", [])}:
        return "tag"
    if meta.get("Status", "").lower() in {s.lower() for s in ex.get("statuses", [])}:
        return "status"
    d = parse_date(meta.get("Date"))
    if d is None:
        return "no_date"
    for start, end in ex.get("date_ranges", []):
        if parse_date(start) <= d <= parse_date(end):
            return "date_range"
    if cjk_ratio(clean_body(body)) < rules.get("min_cjk_ratio", 0.15):
        return "not_chinese"
    return None


def load_posts(paths: list[Path], rules: dict) -> tuple[list[Post], dict[str, str]]:
    posts, excluded = [], {}
    for path in paths:
        meta, body = parse_post(path)
        reason = exclusion_reason(path, meta, body, rules)
        if reason:
            excluded[path.name] = reason
            continue
        posts.append(
            Post(
                path=path,
                slug=meta.get("Slug") or path.stem,
                title=meta.get("Title", ""),
                date=parse_date(meta.get("Date")),
                tags=[t.strip() for t in meta.get("Tags", "").split(",") if t.strip()],
                body=body,
            )
        )
    return posts, excluded


def paragraphs(body: str) -> list[str]:
    blocks = re.split(r"\n\s*\n", clean_body(body))
    return [b.strip() for b in blocks if b.strip() and CJK_RE.search(b)]


def split_long(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    cuts = [m.end() for m in re.finditer(r"[。！？!?]", text)]
    cuts = [c for c in cuts if 0 < c < len(text)]
    if not cuts:
        return [text]
    mid = min(cuts, key=lambda c: abs(c - len(text) / 2))
    return split_long(text[:mid].strip(), limit) + split_long(text[mid:].strip(), limit)


def make_units(post: Post, min_chars: int = 50, max_chars: int = 800) -> list[Unit]:
    merged: list[str] = []
    carry = ""
    for para in paragraphs(post.body):
        para = f"{carry}\n\n{para}" if carry else para
        carry = ""
        if len(para) < min_chars:
            carry = para
            continue
        merged.extend(split_long(para, max_chars))
    if carry:
        if merged:
            merged[-1] = f"{merged[-1]}\n\n{carry}"
        else:
            merged.append(carry)
    return [
        Unit(
            unit_id=f"{post.slug}#{i:03d}",
            slug=post.slug,
            date=post.date.isoformat() if post.date else "",
            index=i,
            text=text,
            prev=merged[i - 1] if i else "",
        )
        for i, text in enumerate(merged)
    ]


def assign_splits(
    posts: list[Post],
    n_test: int = 20,
    n_val: int = 10,
    seed: int = 0,
    min_test_per_bucket: dict[str, int] | None = None,
) -> dict[str, str]:
    """Split by post (never by paragraph), stratified by year; each year with 2+ posts gets a test post."""
    rng = random.Random(seed)
    buckets: dict[str, list[str]] = {}
    for p in sorted(posts, key=lambda p: p.slug):
        buckets.setdefault(str(p.date.year) if p.date else "unknown", []).append(p.slug)
    total = max(1, len(posts))
    splits: dict[str, str] = {}
    for name, slugs in sorted(buckets.items()):
        rng.shuffle(slugs)
        want = max((min_test_per_bucket or {}).get(str(name), 1), round(n_test * len(slugs) / total))
        k_test = min(want, len(slugs) - 1)
        k_val = round(n_val * len(slugs) / total)
        for i, slug in enumerate(slugs):
            splits[slug] = "test" if i < k_test else "val" if i < k_test + k_val else "train"
    return splits
