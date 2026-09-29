"""SFT example format and the mechanical fact filter shared by every dataset builder."""
from __future__ import annotations

import re

NUM_RE = re.compile(r"\d+(?:\.\d+)?")
LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+\-.]*")

# Single-digit and ten: a rewrite may switch between Arabic and Chinese numerals.
CN_DIGIT = {"1": "一", "2": "二两", "3": "三", "4": "四", "5": "五", "6": "六", "7": "七", "8": "八", "9": "九", "10": "十"}


def numbers(text: str) -> set[str]:
    return {n.rstrip(".") for n in NUM_RE.findall(text)}


def latin(text: str) -> set[str]:
    return {w.lower().strip(".-") for w in LATIN_RE.findall(text)}


def number_ok(num: str, other: str) -> bool:
    """A small number may legitimately switch between Arabic and Chinese numerals."""
    return num in CN_DIGIT and any(ch in other for ch in CN_DIGIT[num])


def fact_diff(original: str, rewrite: str) -> tuple[set, set]:
    """Numbers that were dropped or invented, and Latin tokens the rewrite invented.

    Dropped Latin tokens are allowed: AI rewrites often translate English terms into
    Chinese, and learning to undo that is part of the target style.
    """
    num_o, num_r = numbers(original), numbers(rewrite)
    bad_nums = {n for n in num_o - num_r if not number_ok(n, rewrite)}
    bad_nums |= {n for n in num_r - num_o if not number_ok(n, original)}
    lat_o = latin(original)
    invented = {w for w in latin(rewrite) - lat_o if not any(w in x or x in w for x in lat_o)}
    return bad_nums, invented


def keep_rewrite(original: str, rewrite: str, length_ratio: tuple[float, float] = (0.5, 1.9), max_invented_latin: int = 1) -> tuple[bool, str]:
    """Whether an AI rewrite is faithful enough to train on, and why not."""
    if not rewrite.strip():
        return False, "empty"
    ratio = len(rewrite) / max(1, len(original))
    if not length_ratio[0] <= ratio <= length_ratio[1]:
        return False, "length_ratio"
    bad_nums, invented = fact_diff(original, rewrite)
    if bad_nums:
        return False, "number_changed"
    if len(invented) > max_invented_latin:
        return False, "latin_invented"
    return True, "ok"


def user_message(instruction: str, ai_text: str, prev: str = "") -> str:
    """The prompt the style model sees, at training and at inference. Keep both sides identical."""
    parts = [instruction]
    if prev:
        parts.append(f"【上文，仅供参考，不要改写】\n{prev}")
    parts.append(f"【要改写的段落】\n{ai_text}")
    return "\n\n".join(parts)


def to_example(instruction: str, unit: dict, ai_text: str) -> dict:
    return {
        "unit_id": unit["unit_id"],
        "messages": [
            {"role": "user", "content": user_message(instruction, ai_text, unit.get("prev", ""))},
            {"role": "assistant", "content": unit["text"]},
        ],
    }
