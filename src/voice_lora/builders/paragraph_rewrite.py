"""v1 dataset builder: rewrite each paragraph of the author's text into AI-flavoured Chinese with several
prompts, then train on the reverse direction (AI rewrite -> original).

Each prompt produces a different kind of AI voice, so the model learns to undo more than one of them.
The rewrites keep the original's order and argument, so the model learns wording and sentence-level
voice only; a structure-level builder (outline_regen) was tried and not adopted, see docs/v2_experiment.md.
"""
from __future__ import annotations

import hashlib
import random
import re
from collections import Counter
from pathlib import Path

import httpx

from ..config import Config
from ..models import ModelPool, base_prompt, model_specs
from ..pairs import keep_rewrite, to_example

COMMON_RULES = (
    "要求：保留原文的全部信息、观点和先后顺序，不增加原文没有的事实、数字或例子，不删掉任何论点；"
    "如果原文是列表，保持列表形式。只输出改写后的正文，不要标题、不要解释、不要引号。"
)

PROMPTS = {
    "default": "在不改变顺序和逻辑的前提下，用你自己最舒适、最自然的语言风格重写下面这段文字。",
    "formal": "把下面这段文字改写成适合正式发表的书面文章段落，措辞规范、表达严谨。",
    "style_mimic": "下面先给出两段示例文字，请模仿示例的文风（用词、句式、节奏），重写后面的目标段落。",
    "expand": "对下面这段文字进行润色，让表达更完整、更流畅，补足句子之间的过渡和衔接，使读者更容易理解。",
    "condense": "把下面这段文字改写得更简洁、更专业，去掉口语化的表达。",
}

# Which rewrite to use as the evaluation input for a test paragraph, in order of preference.
# style_mimic is closest to the AI articles the model will actually be asked to fix.
EVAL_PREFERENCE = ["style_mimic", "default", "formal", "expand", "condense"]

CJK_RE = re.compile(r"[一-鿿]")


def load_style_pool(cfg: Config, min_len: int = 150, max_len: int = 400) -> list[str]:
    """Prose paragraphs from AI-written articles, used as style exemplars for the style_mimic prompt."""
    section = cfg.get("builder.style_exemplars", {}) or {}
    excluded = {Path(x).name for x in section.get("exclude", [])}
    excluded |= {Path(x).name for x in cfg.get("evaluate.articles", []) or []}
    pool = []
    for path in cfg.glob(section.get("paths", [])):
        if path.name in excluded:
            continue
        for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8", errors="replace")):
            block = block.strip()
            if block.startswith(("#", "!", "|", ">", "```", "<", "-", "*")):
                continue
            if min_len <= len(block) <= max_len and len(CJK_RE.findall(block)) > 0.6 * len(block):
                pool.append(re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", block))
    return pool


def build_messages(unit: dict, prompt_id: str, style_pool: list[str]) -> list[dict]:
    parts = [PROMPTS[prompt_id], COMMON_RULES]
    if prompt_id == "style_mimic":
        if len(style_pool) < 2:
            raise ValueError("style_mimic needs at least two exemplar paragraphs; check builder.style_exemplars.paths")
        rng = random.Random(hashlib.md5(f"{unit['unit_id']}|style_mimic".encode()).hexdigest())
        a, b = rng.sample(style_pool, 2)
        parts.append(f"【示例一】\n{a}\n\n【示例二】\n{b}")
    if unit.get("prev"):
        parts.append(f"【上文，仅帮助理解语境，不要改写、不要输出】\n{unit['prev']}")
    parts.append(f"【需要改写的段落】\n{unit['text']}")
    return [{"role": "user", "content": "\n\n".join(parts)}]


def rewriters(cfg: Config) -> list[dict]:
    """`rewriters` (a list of models, each with its own prompts) or the single `rewriter` with `builder.prompts`.
    The first model's rewrites keep plain prompt ids; later models get "@name" (see voice_lora.models)."""
    return model_specs(cfg, "rewriters", "rewriter", cfg.get("builder.prompts", list(PROMPTS)), PROMPTS)


def make_jobs(cfg: Config, units: list[dict], done: set[str]) -> list[dict]:
    jobs = []
    for m in rewriters(cfg):
        for u in units:
            for p in m["prompts"]:
                pid = p + m["suffix"]
                if f"{u['unit_id']}|{pid}" not in done:
                    jobs.append({"unit_id": u["unit_id"], "prompt_id": pid, "model": m["name"], "unit": u})
    return jobs


def make_worker(cfg: Config, _client, style_pool: list[str]):
    pool = ModelPool(cfg, rewriters(cfg))

    async def worker(http: httpx.AsyncClient, job: dict) -> dict:
        unit = job["unit"]
        messages = build_messages(unit, base_prompt(job["prompt_id"]), style_pool)
        res = await pool.complete(http, job["model"], messages, max_tokens=min(2048, 3 * len(unit["text"]) + 200),
                                  job_id=f"{unit['unit_id']}_{job['prompt_id']}")
        return {"unit_id": unit["unit_id"], "prompt_id": job["prompt_id"], "rewrite": res["text"],
                "finish_reason": res.get("finish_reason"), "usage": res.get("usage", {}), "source": res["source"]}

    worker.concurrency = pool.concurrency  # type: ignore[attr-defined]
    return worker


def build(cfg: Config, units: list[dict], rewrites: list[dict]) -> tuple[dict[str, list[dict]], list[dict], Counter]:
    """Filter rewrites and turn them into SFT examples per split, plus one evaluation input per test unit."""
    by_id = {u["unit_id"]: u for u in units}
    flt = cfg.get("builder.filter", {}) or {}
    ratio = tuple(flt.get("length_ratio", [0.5, 1.9]))
    max_latin = int(flt.get("max_invented_latin", 1))
    seen: set[str] = set()
    kept: dict[str, list[tuple[dict, dict]]] = {"train": [], "val": [], "test": []}
    reasons: Counter = Counter()
    for r in rewrites:
        key = f"{r['unit_id']}|{r['prompt_id']}"
        if key in seen or not r.get("rewrite") or r["unit_id"] not in by_id:
            reasons["missing_or_duplicate"] += 1
            continue
        seen.add(key)
        unit = by_id[r["unit_id"]]
        ok, why = keep_rewrite(unit["text"], r["rewrite"], ratio, max_latin)
        reasons[f"{r['prompt_id']}:{why}"] += 1
        if ok:
            kept[unit["split"]].append((unit, r))

    instruction = cfg.instruction
    rng = random.Random(int(cfg.get("corpus.split.seed", 0)))
    identity_fraction = float(cfg.get("builder.identity_fraction", 0.05))
    examples: dict[str, list[dict]] = {}
    for split in ("train", "val"):
        ex = [to_example(instruction, u, r["rewrite"]) for u, r in kept[split]]
        split_units = [u for u in units if u["split"] == split]
        identity = rng.sample(split_units, int(len(split_units) * identity_fraction))
        ex += [to_example(instruction, u, u["text"]) for u in identity]
        rng.shuffle(ex)
        examples[split] = ex

    by_unit: dict[str, dict[str, dict]] = {}
    for u, r in kept["test"]:
        by_unit.setdefault(u["unit_id"], {})[r["prompt_id"]] = r
    eval_inputs = []
    for uid in sorted(by_unit):
        options = by_unit[uid]
        ranked = sorted(options, key=lambda pid: ("@" in pid, EVAL_PREFERENCE.index(base_prompt(pid)) if base_prompt(pid) in EVAL_PREFERENCE else 99))
        r = options[ranked[0]]
        u = by_id[uid]
        eval_inputs.append({"unit_id": uid, "prompt_id": r["prompt_id"], "ai_text": r["rewrite"], "prev": u["prev"], "original": u["text"]})
    return examples, eval_inputs, reasons
