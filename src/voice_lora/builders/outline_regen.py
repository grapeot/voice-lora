"""v2 dataset builder: structure-level voice.

Cut each of the author's articles into sections (at headings, then windows of a few paragraphs), extract an
outline plus a fact list from each section (stage `extract`, any fast model), have an AI write that section
from the outline alone (stage `draft`, ideally the same kind of model that drafts the articles you will
rewrite later), and train on (AI section -> original section). Unlike paragraph_rewrite, the AI side owns
the paragraphing and the argument flow inside the section, as in real AI-drafted articles.

Facts are the risk: a draft that adds numbers or names teaches the model to delete facts, and one that
drops any of the original's numbers teaches it to invent them (the target then holds numbers the input
lacks). `build` filters both ways; list numbering ("1) ... 2) ...") is not counted as a fact.
"""
from __future__ import annotations

import hashlib
import random
import re
from collections import Counter

import httpx

from ..config import Config
from ..corpus import Post, Unit, make_sections
from ..llm import ChatClient, make_client
from ..metrics import GROUPED_NUM_RE
from ..pairs import latin, to_example
from .paragraph_rewrite import load_style_pool  # noqa: F401  (re-exported for 02_generate)

STAGES = ["extract", "draft"]

EXTRACT_PROMPT = (
    "阅读下面这一节文章。先按原文顺序列出它的要点，每个要点一行，只概括意思，不要照抄原句；"
    "再列出其中出现的全部具体事实：数字、人名、机构名、产品名、书名、例子、引语，一项一行，照原文写。\n"
    "只输出下面的格式，不要别的内容：\n要点：\n1. ……\n事实：\n- ……"
)

DRAFT_PROMPTS = {
    "draft_default": "下面是一节文章的要点和事实清单。请据此写出这一节的正文，约 {n} 字。用你自己最自然的写法，自己决定怎么分段、怎么组织。",
    "draft_style": (
        "下面先给出两段示例文字，然后是一节文章的要点和事实清单。"
        "请模仿示例的文风（用词、句式、节奏），据此写出这一节的正文，约 {n} 字，自己决定怎么分段、怎么组织。"
    ),
    "draft_formal": "下面是一节文章的要点和事实清单。请据此写出这一节适合正式发表的正文，约 {n} 字，措辞规范、表达严谨，自己决定怎么分段、怎么组织。",
}
DRAFT_RULES = "要求：只能使用清单里的事实，不要增加清单以外的数字、人名、机构或例子；要点都要写到；不要标题，不要解释，只输出正文。"

EVAL_PREFERENCE = ["draft_style", "draft_default", "draft_formal"]


def make_units(post: Post, cfg: Config) -> list[Unit]:
    s = cfg.get("builder.sections", {}) or {}
    return make_sections(post, s.get("target_chars", 900), s.get("max_paras", 6), s.get("min_tail", 300))


# "1) ... 2) ..." or "1. ... 2. ..." enumerations: a draft that turns a list into prose drops these
# markers without dropping any fact, so they do not count as numbers that must survive.
LIST_MARKER_RE = re.compile(r"(?:^|(?<=[\s:：；;。，,]))\d{1,2}\s*[.)）、](?!\d)", re.M)


def _numbers(text: str, skip_list_markers: bool = False) -> set[str]:
    if skip_list_markers:
        text = LIST_MARKER_RE.sub(" ", text)
    return {f"{float(n.replace(',', '')):g}" for n in GROUPED_NUM_RE.findall(text)}


def drafters(cfg: Config) -> list[dict]:
    """The drafting models: `drafters` (a list, each with a `name`) or a single `drafter` section.
    Drafts from the first drafter keep plain prompt ids (draft_style); later ones get "@name" (draft_style@deepseek)."""
    if cfg.get("drafters"):
        items = [dict(d) for d in cfg.get("drafters")]
    else:
        items = [{"name": "drafter", **(cfg.get("drafter") or cfg["rewriter"])}]
    default_prompts = cfg.get("builder.prompts", list(DRAFT_PROMPTS))
    for i, d in enumerate(items):
        d["suffix"] = "" if i == 0 else f"@{d['name']}"
        d.setdefault("prompts", default_prompts)
        unknown = [p for p in d["prompts"] if p not in DRAFT_PROMPTS]
        if unknown:
            raise ValueError(f"unknown draft prompts {unknown}; choose from {list(DRAFT_PROMPTS)}")
    return items


def base_prompt(prompt_id: str) -> str:
    return prompt_id.split("@", 1)[0]


def make_jobs(cfg: Config, units: list[dict], done: set[str], stage: str = "draft", outlines: dict[str, str] | None = None) -> list[dict]:
    if stage == "extract":
        return [{"unit_id": u["unit_id"], "prompt_id": "extract", "unit": u} for u in units if f"{u['unit_id']}|extract" not in done]
    if stage != "draft":
        raise ValueError(f"unknown stage {stage!r}; outline_regen has {STAGES}")
    outlines = outlines or {}
    missing = [u["unit_id"] for u in units if u["unit_id"] not in outlines]
    if missing:
        raise ValueError(f"{len(missing)} sections have no outline yet (e.g. {missing[0]}); run --stage extract first")
    jobs = []
    for d in drafters(cfg):
        for u in units:
            for p in d["prompts"]:
                pid = p + d["suffix"]
                if f"{u['unit_id']}|{pid}" not in done:
                    jobs.append({"unit_id": u["unit_id"], "prompt_id": pid, "drafter": d["name"], "unit": u, "outline": outlines[u["unit_id"]]})
    return jobs


def build_draft_messages(unit: dict, prompt_id: str, outline: str, style_pool: list[str]) -> list[dict]:
    prompt_id = base_prompt(prompt_id)
    parts = [DRAFT_PROMPTS[prompt_id].format(n=len(unit["text"])), DRAFT_RULES]
    if prompt_id == "draft_style":
        if len(style_pool) < 2:
            raise ValueError("draft_style needs style exemplars; check builder.style_exemplars.paths")
        rng = random.Random(hashlib.md5(f"{unit['unit_id']}|draft_style".encode()).hexdigest())
        a, b = rng.sample(style_pool, 2)
        parts.append(f"【示例一】\n{a}\n\n【示例二】\n{b}")
    parts.append(f"【要点与事实】\n{outline}")
    return [{"role": "user", "content": "\n\n".join(parts)}]


def make_worker(cfg: Config, _client_unused, style_pool: list[str], stage: str = "draft"):
    """`extract` uses the `rewriter` section; `draft` uses each drafter's own client and concurrency."""
    import asyncio

    if stage == "extract":
        clients = {"extract": ChatClient(cfg["rewriter"])}
    else:
        clients = {}
        for d in drafters(cfg):
            run_root = cfg.path(d["run_dir"]) if d.get("run_dir") else cfg.work("command_runs", "x").parent
            clients[d["name"]] = make_client(d, run_root)
    limits = {name: asyncio.Semaphore(c.concurrency) for name, c in clients.items()}

    async def worker(http: httpx.AsyncClient, job: dict) -> dict:
        unit = job["unit"]
        name = "extract" if stage == "extract" else job["drafter"]
        client = clients[name]
        if stage == "extract":
            messages = [{"role": "user", "content": f"{EXTRACT_PROMPT}\n\n【文章】\n{unit['text']}"}]
        else:
            messages = build_draft_messages(unit, job["prompt_id"], job["outline"], style_pool)
        kwargs = {"job_id": f"{unit['unit_id']}_{job['prompt_id']}"} if not isinstance(client, ChatClient) else {}
        async with limits[name]:
            res = await client.complete(http, messages, max_tokens=min(4096, 3 * len(unit["text"]) + 400), **kwargs)
        key = "outline" if stage == "extract" else "rewrite"
        return {"unit_id": unit["unit_id"], "prompt_id": job["prompt_id"], key: res["text"], "usage": res.get("usage", {}),
                "source": getattr(client, "model", None) or "command"}

    worker.concurrency = sum(c.concurrency for c in clients.values())  # type: ignore[attr-defined]
    return worker


def keep_draft(original: str, draft: str, ratio_range=(0.6, 1.7), min_number_coverage: float = 1.0, max_invented_latin: int = 1) -> tuple[bool, str]:
    if not draft.strip():
        return False, "empty"
    ratio = len(draft) / max(1, len(original))
    if not ratio_range[0] <= ratio <= ratio_range[1]:
        return False, "length_ratio"
    no, nd = _numbers(original), _numbers(draft)
    if nd - no:
        return False, "number_invented"
    facts = _numbers(original, skip_list_markers=True)
    if facts and len(facts & nd) / len(facts) < min_number_coverage:
        return False, "numbers_missing"
    lo = latin(original)
    invented = {w for w in latin(draft) - lo if not any(w in x or x in w for x in lo)}
    if len(invented) > max_invented_latin:
        return False, "latin_invented"
    return True, "ok"


def build(cfg: Config, units: list[dict], rewrites: list[dict]) -> tuple[dict[str, list[dict]], list[dict], Counter]:
    by_id = {u["unit_id"]: u for u in units}
    flt = cfg.get("builder.filter", {}) or {}
    ratio = tuple(flt.get("length_ratio", [0.6, 1.7]))
    coverage = float(flt.get("min_number_coverage", 1.0))
    max_latin = int(flt.get("max_invented_latin", 1))
    kept: dict[str, list[tuple[dict, dict]]] = {"train": [], "val": [], "test": []}
    seen: set[str] = set()
    reasons: Counter = Counter()
    for r in rewrites:
        key = f"{r['unit_id']}|{r['prompt_id']}"
        if r.get("prompt_id") == "extract" or key in seen or not r.get("rewrite") or r["unit_id"] not in by_id:
            continue
        seen.add(key)
        unit = by_id[r["unit_id"]]
        ok, why = keep_draft(unit["text"], r["rewrite"], ratio, coverage, max_latin)
        reasons[f"{r['prompt_id']}:{why}"] += 1
        if ok:
            kept[unit["split"]].append((unit, r))
    instruction = cfg.instruction
    rng = random.Random(int(cfg.get("corpus.split.seed", 0)))
    examples = {}
    for split in ("train", "val"):
        ex = [to_example(instruction, u, r["rewrite"]) for u, r in kept[split]]
        rng.shuffle(ex)
        examples[split] = ex
    by_unit: dict[str, dict[str, dict]] = {}
    for u, r in kept["test"]:
        by_unit.setdefault(u["unit_id"], {})[r["prompt_id"]] = r
    eval_inputs = []
    for uid in sorted(by_unit):
        options = by_unit[uid]
        r = next((options[p] for p in EVAL_PREFERENCE if p in options), next(iter(options.values())))  # first drafter first
        u = by_id[uid]
        eval_inputs.append({"unit_id": uid, "prompt_id": r["prompt_id"], "ai_text": r["rewrite"], "prev": u["prev"], "original": u["text"]})
    return examples, eval_inputs, reasons

