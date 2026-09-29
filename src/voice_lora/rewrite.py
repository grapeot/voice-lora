"""Call a trained voice model behind an OpenAI-compatible server (LM Studio, vLLM) and rewrite articles with it.

A model card says how to call one deployed model: where it is served, its identifier there, and the
instruction it was trained with (the prompt must match training). The sampling defaults are part of
calling it correctly, so every caller should go through `request_body` instead of building requests:
- greedy decoding mostly copies the input (16 of 40 paragraphs came back unchanged in one article);
- LM Studio ignores vLLM's `repetition_penalty` key and silently applies its own default, so both keys are sent.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

from .articles import guarded, rewrite_article, save
from .config import Config
from .pairs import user_message

DEFAULT_SAMPLING = {"temperature": 0.7, "top_p": 0.95, "repetition_penalty": 1.05}


@dataclass
class ModelCard:
    name: str
    base_url: str
    model: str
    instruction: str
    sampling: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_SAMPLING))

    @classmethod
    def load(cls, path: str | Path) -> ModelCard:
        raw = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8")) or {}
        missing = [k for k in ("base_url", "model", "instruction") if not raw.get(k)]
        if missing:
            raise ValueError(f"model card {path} lacks {missing}")
        return cls(name=raw.get("name") or raw["model"], base_url=raw["base_url"], model=raw["model"],
                   instruction=raw["instruction"], sampling={**DEFAULT_SAMPLING, **(raw.get("sampling") or {})})

    @classmethod
    def from_config(cls, cfg: Config, base_url: str | None = None, model: str | None = None, name: str | None = None) -> ModelCard:
        """A card from the training config (its `serve` section and author instruction), with overrides."""
        serve = cfg.get("serve", {}) or {}
        base_url = base_url or serve.get("base_url")
        model = model or serve.get("model", "voice-lora")
        if not base_url:
            raise ValueError("no server: give --serve-url or set serve.base_url in the config")
        return cls(name=name or model, base_url=base_url, model=model, instruction=cfg.instruction)

    def to_yaml(self) -> str:
        return yaml.safe_dump({"name": self.name, "base_url": self.base_url, "model": self.model,
                               "instruction": self.instruction, "sampling": self.sampling}, allow_unicode=True, sort_keys=False)


def sampling_overrides(temperature: float | None = None, repetition_penalty: float | None = None) -> dict[str, Any]:
    """Command-line overrides; temperature 0 (greedy, for checking that two deployments agree) also drops top_p."""
    out: dict[str, Any] = {}
    if temperature is not None:
        out["temperature"] = temperature
        if temperature == 0:
            out["top_p"] = 1.0
    if repetition_penalty is not None:
        out["repetition_penalty"] = repetition_penalty
    return out


def request_body(card: ModelCard, text: str, prev: str = "", sampling: dict[str, Any] | None = None) -> dict[str, Any]:
    s = {**DEFAULT_SAMPLING, **card.sampling, **(sampling or {})}
    penalty = s.pop("repetition_penalty")
    return {"model": card.model, "messages": [{"role": "user", "content": user_message(card.instruction, text, prev)}],
            **s, "repetition_penalty": penalty, "repeat_penalty": penalty,
            "max_tokens": min(2048, int(len(text) * 1.6) + 64), "chat_template_kwargs": {"enable_thinking": False}}


class Rewriter:
    """`await rewriter(text, prev)` -> rewritten text. Counts completion tokens in `tokens`."""

    def __init__(self, card: ModelCard, http: httpx.AsyncClient, concurrency: int = 4, sampling: dict[str, Any] | None = None):
        self.card, self.http, self.sampling = card, http, sampling
        self.url = card.base_url.rstrip("/") + "/chat/completions"
        self.sem = asyncio.Semaphore(concurrency)
        self.tokens = 0

    async def __call__(self, text: str, prev: str = "") -> str:
        async with self.sem:
            r = await self.http.post(self.url, json=request_body(self.card, text, prev, self.sampling))
        r.raise_for_status()
        d = r.json()
        self.tokens += d.get("usage", {}).get("completion_tokens", 0)
        return (d["choices"][0]["message"]["content"] or "").replace("<think>", "").replace("</think>", "").strip()


async def rewrite_files(card: ModelCard, jobs: list[tuple[Path, Path]], concurrency: int = 4, unit: str = "paragraph",
                        mode: str = "parallel", guard: bool = True, sampling: dict[str, Any] | None = None,
                        target_chars: int = 900, max_paras: int = 6, transport: httpx.AsyncBaseTransport | None = None) -> list[dict]:
    """Rewrite each (source, output) markdown pair; writes output.md plus output.blocks.jsonl (the per-unit log).

    With `guard`, a unit whose numbers change or whose length ratio leaves 0.6-1.6 is re-sampled once and then
    kept as the input; those units are listed under "kept_original"."""
    results = []
    async with httpx.AsyncClient(timeout=600, transport=transport) as http:
        rw = Rewriter(card, http, concurrency, sampling)
        for src, out in jobs:
            fn = guarded(rw) if guard else rw
            before = rw.tokens
            md, log = await rewrite_article(src, fn, mode=mode, unit=unit, target_chars=target_chars, max_paras=max_paras)
            save(md, log, out)
            results.append({"source": str(src), "out": str(out), "units": len(log), "tokens": rw.tokens - before,
                            "kept_original": list(getattr(fn, "guarded_log", []))})
    return results
