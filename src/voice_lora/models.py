"""Several generating models in one run: each model has its own client, concurrency and prompt list.

Outputs from the first model keep plain prompt ids (`style_mimic`); later models get an "@name" suffix
(`style_mimic@deepseek`), so one unit can hold candidates from several model families side by side.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from .config import Config
from .llm import ChatClient, make_client


def model_specs(cfg: Config, list_key: str, single_key: str, default_prompts: list[str], known_prompts: dict) -> list[dict]:
    """The models under `list_key` (a list, each with a `name`), or a single section under `single_key`."""
    if cfg.get(list_key):
        items = [dict(m) for m in cfg.get(list_key)]
    else:
        section = cfg.get(single_key)
        if not section:
            raise ValueError(f"config needs `{list_key}` or `{single_key}`")
        items = [{"name": single_key, **section}]
    for i, m in enumerate(items):
        m["suffix"] = "" if i == 0 else f"@{m['name']}"
        m.setdefault("prompts", default_prompts)
        unknown = [p for p in m["prompts"] if p not in known_prompts]
        if unknown:
            raise ValueError(f"unknown prompts {unknown} for {m['name']}; choose from {list(known_prompts)}")
    return items


def base_prompt(prompt_id: str) -> str:
    return prompt_id.split("@", 1)[0]


class ModelPool:
    """One client and one concurrency limit per model, so a slow provider does not hold up a fast one."""

    def __init__(self, cfg: Config, specs: list[dict]):
        self.clients = {}
        for m in specs:
            run_root = cfg.path(m["run_dir"]) if m.get("run_dir") else cfg.work("command_runs", "x").parent
            self.clients[m["name"]] = make_client(m, Path(run_root))
        self.limits = {name: asyncio.Semaphore(c.concurrency) for name, c in self.clients.items()}
        self.concurrency = sum(c.concurrency for c in self.clients.values())

    async def complete(self, http, name: str, messages: list[dict], max_tokens: int, job_id: str = "") -> dict:
        client = self.clients[name]
        kwargs = {} if isinstance(client, ChatClient) else {"job_id": job_id}
        async with self.limits[name]:
            res = await client.complete(http, messages, max_tokens=max_tokens, **kwargs)
        return {**res, "source": getattr(client, "model", None) or "command"}
