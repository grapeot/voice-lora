#!/usr/bin/env python3
"""Baseline to beat: the rewriter model with k retrieved paragraphs of the author's own writing as examples.
If a LoRA cannot beat this, prompting is enough and the fine-tune is not worth it.

Writes <workdir>/eval/baseline_fewshot.jsonl and, for each evaluate.articles entry,
<workdir>/eval/articles/<stem>.baseline_fewshot.md (+ .blocks.jsonl).
"""
import asyncio

import httpx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import _bootstrap  # noqa: F401
from voice_lora import jsonl
from voice_lora.articles import rewrite_article, save
from voice_lora.cli import config_from, parser
from voice_lora.llm import ChatClient


class Retriever:
    def __init__(self, docs: list[str]):
        self.docs = docs
        self.vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), sublinear_tf=True)
        self.mat = self.vec.fit_transform(docs)

    def top(self, query: str, k: int) -> list[str]:
        sims = linear_kernel(self.vec.transform([query]), self.mat).ravel()
        return [self.docs[i] for i in sims.argsort()[::-1][:k]]


def prompt(name: str, ai_text: str, prev: str, examples: list[str]) -> str:
    ex = "\n\n".join(f"【例{i + 1}】\n{e}" for i, e in enumerate(examples))
    parts = [f"下面是{name}写的几段原文。先仔细体会他的文风：用词、句式、语气、节奏、口语化程度。", ex,
             f"然后把后面的目标段落改写成{name}的文风。要求：保留全部信息、观点和先后顺序，不增加原文没有的事实，"
             "不要照抄例子里的内容。只输出改写后的正文，不要解释。"]
    if prev:
        parts.append(f"【上文，仅供参考，不要改写】\n{prev}")
    parts.append(f"【目标段落】\n{ai_text}")
    return "\n\n".join(parts)


async def run(cfg) -> None:
    name = cfg.get("author.name", "作者")
    k = int(cfg.get("evaluate.fewshot_k", 8))
    units = jsonl.read(cfg.work("units.jsonl"))
    retriever = Retriever([u["text"] for u in units if u["split"] == "train"])
    client = ChatClient(cfg["rewriter"])
    sem = asyncio.Semaphore(client.concurrency)
    async with httpx.AsyncClient(timeout=600) as http:
        async def rewrite(text: str, prev: str) -> str:
            async with sem:
                res = await client.complete(http, [{"role": "user", "content": prompt(name, text, prev, retriever.top(text, k))}])
                return res["text"]

        rows = jsonl.read(cfg.work("eval_inputs.jsonl"))
        outs = await asyncio.gather(*(rewrite(r["ai_text"], r.get("prev", "")) for r in rows))
        jsonl.write(cfg.work("eval", "baseline_fewshot.jsonl"), [{**r, "output": o, "system": "baseline_fewshot"} for r, o in zip(rows, outs, strict=True)])
        for a in cfg.get("evaluate.articles", []) or []:
            src = cfg.path(a)
            md, log = await rewrite_article(src, rewrite, mode="parallel")
            save(md, log, cfg.work("eval", "articles", f"{src.stem}.baseline_fewshot.md"))
    print("done")


def main() -> None:
    asyncio.run(run(config_from(parser(__doc__).parse_args())))


if __name__ == "__main__":
    main()
