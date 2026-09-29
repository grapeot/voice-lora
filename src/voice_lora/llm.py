"""Minimal async client for OpenAI-compatible chat endpoints, with bounded concurrency and resumable jsonl output."""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx


class ChatClient:
    def __init__(self, section: dict[str, Any]):
        self.base_url = section["base_url"].rstrip("/")
        self.model = section["model"]
        key_env = section.get("api_key_env")
        self.api_key = os.environ.get(key_env, "") if key_env else ""
        self.sampling = dict(section.get("sampling", {}))
        self.extra_body = dict(section.get("extra_body", {}))
        self.concurrency = int(section.get("concurrency", 4))

    def body(self, messages: list[dict], max_tokens: int) -> dict:
        return {"model": self.model, "messages": messages, "max_tokens": max_tokens, **self.sampling, **self.extra_body}

    async def complete(self, client: httpx.AsyncClient, messages: list[dict], max_tokens: int = 2048, retries: int = 3) -> dict:
        """Returns {"text", "usage", "finish_reason"}; raises the last error after `retries` attempts."""
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        last: Exception | None = None
        for attempt in range(retries):
            try:
                r = await client.post(f"{self.base_url}/chat/completions", json=self.body(messages, max_tokens), headers=headers)
                r.raise_for_status()
                d = r.json()
                choice = d["choices"][0]
                return {
                    "text": clean_output(choice["message"].get("content") or ""),
                    "usage": d.get("usage", {}),
                    "finish_reason": choice.get("finish_reason"),
                }
            except (httpx.HTTPError, KeyError, ValueError) as e:
                last = e
                await asyncio.sleep(5 * (attempt + 1))
        raise RuntimeError(f"request failed after {retries} attempts: {last}")


def clean_output(text: str) -> str:
    """Strip reasoning blocks and labels some models prepend to a rewrite."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^【?(改写后|改写结果|重写后)[^】\n]*】?[:：]?\s*", "", text)
    return text.strip().strip('"“”')


def done_keys(path: Path, key: Callable[[dict], str]) -> set[str]:
    keys: set[str] = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                if not rec.get("error"):
                    keys.add(key(rec))
    return keys


async def run_jobs(
    jobs: list[dict],
    worker: Callable[[httpx.AsyncClient, dict], Awaitable[dict]],
    out_path: Path,
    concurrency: int,
    progress_every: int = 25,
) -> None:
    """Run `worker` over `jobs` with bounded concurrency, appending one json line per finished job.
    Failed jobs are written with an "error" field and retried on the next run."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    t0, done, tokens = time.time(), 0, 0
    async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=10)) as client:
        with out_path.open("a") as f:

            async def one(job: dict) -> None:
                nonlocal done, tokens
                async with sem:
                    try:
                        rec = await worker(client, job)
                    except Exception as e:  # noqa: BLE001 - recorded and retried on the next run
                        rec = {**{k: v for k, v in job.items() if k.endswith("_id")}, "error": str(e)}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                done += 1
                tokens += rec.get("usage", {}).get("completion_tokens", 0)
                if done % progress_every == 0 or done == len(jobs):
                    el = time.time() - t0
                    print(f"{done}/{len(jobs)}  {el / 60:.1f} min  {tokens / max(el, 1e-6):.0f} out tok/s", flush=True)

            await asyncio.gather(*(one(j) for j in jobs))
