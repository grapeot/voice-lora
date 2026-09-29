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
        last = ""
        for attempt in range(retries):
            try:
                r = await client.post(f"{self.base_url}/chat/completions", json=self.body(messages, max_tokens), headers=headers)
            except httpx.HTTPError as e:
                last = f"{type(e).__name__}: {e}"
                await asyncio.sleep(5 * (attempt + 1))
                continue
            # Keep the provider's own words: status, finish_reason and a slice of the body say why a call
            # failed (rate limit, content filter, malformed request) far better than a bare KeyError.
            body = r.text[:500]
            if r.status_code != 200:
                last = f"HTTP {r.status_code}: {body}"
            else:
                try:
                    d = r.json()
                    choice = d["choices"][0]
                    content = (choice.get("message") or {}).get("content")
                    if content:
                        return {"text": clean_output(content), "usage": d.get("usage", {}), "finish_reason": choice.get("finish_reason")}
                    last = f"no content (finish_reason={choice.get('finish_reason')!r}): {body}"
                except (ValueError, KeyError, IndexError, TypeError) as e:
                    last = f"unexpected response ({type(e).__name__}: {e}): {body}"
            await asyncio.sleep(5 * (attempt + 1))
        raise RuntimeError(f"request failed after {retries} attempts: {last}")


class CommandClient:
    """Runs a CLI agent once per request, for writers that are only available as a command line tool.

    Config (the `kind: command` form of a client section):
      command: ["some-cli", "--print", "Read {prompt_file} and follow it.", "--model", "x"]
      output_file: output.txt        # the prompt asks the tool to write its answer here
      prompt_suffix: "把结果写入 {output_file}，只写正文。"
      timeout_s: 600
      concurrency: 10
      run_dir: /path/to/scratch     # optional; default <workdir>/command_runs
    Each request gets its own empty directory (used as cwd) under the run directory, so tools that
    pick up project context from their working directory start clean. {dir}, {prompt_file} and
    {output_file} are substituted in `command` and `prompt_suffix`.
    """

    def __init__(self, section: dict[str, Any], run_root: Path):
        self.command = list(section["command"])
        self.output_name = section.get("output_file", "output.txt")
        self.prompt_suffix = section.get("prompt_suffix", "")
        self.timeout_s = int(section.get("timeout_s", 600))
        self.concurrency = int(section.get("concurrency", 4))
        self.run_root = run_root

    async def complete(self, _http, messages: list[dict], max_tokens: int = 2048, retries: int = 1, job_id: str = "") -> dict:
        prompt = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
        last = ""
        for attempt in range(retries + 1):
            d = self.run_root / re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{job_id or 'job'}_{attempt}_{time.time_ns()}")
            d.mkdir(parents=True, exist_ok=True)
            prompt_file, output_file = d / "prompt.md", d / self.output_name
            fill = {"dir": str(d), "prompt_file": str(prompt_file), "output_file": str(output_file)}
            prompt_file.write_text(prompt + ("\n\n" + self.prompt_suffix.format(**fill) if self.prompt_suffix else ""), encoding="utf-8")
            cmd = [c.format(**fill) for c in self.command]
            proc = await asyncio.create_subprocess_exec(*cmd, cwd=d, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            try:
                out, err = await asyncio.wait_for(proc.communicate(), timeout=self.timeout_s)
            except asyncio.TimeoutError:
                proc.kill()
                last = f"timeout after {self.timeout_s}s"
                continue
            (d / "stdout.txt").write_bytes(out)
            (d / "stderr.txt").write_bytes(err)
            text = output_file.read_text(encoding="utf-8") if output_file.exists() else ""
            if proc.returncode == 0 and text.strip():
                return {"text": clean_output(text), "usage": {}, "finish_reason": "stop"}
            last = f"exit {proc.returncode}, output {'missing' if not text.strip() else 'ok'}: {err.decode(errors='replace')[-300:]}"
        raise RuntimeError(f"command failed: {last}")


def make_client(section: dict[str, Any], run_root: Path):
    return CommandClient(section, run_root) if section.get("kind") == "command" else ChatClient(section)


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
