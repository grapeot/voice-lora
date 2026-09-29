import asyncio
import sys
from datetime import date
from pathlib import Path

from voice_lora.corpus import Post, make_sections
from voice_lora.experimental import outline_regen as orr
from voice_lora.llm import CommandClient

P = "这是一段用来凑长度的中文文字，内容本身没有意义。" * 6  # ~150 chars


def test_sections_split_at_headings_and_window():
    body = f"{P}\n\n{P}\n\n## 第二部分\n\n" + "\n\n".join([P] * 8) + "\n\n短尾巴。"
    secs = make_sections(Post(Path("x.md"), "x", "", date(2020, 1, 1), [], body), target_chars=500, max_paras=6, min_tail=100)
    assert secs[0].text == f"{P}\n\n{P}"                       # the heading closes the first section
    assert all(len(s.text.split("\n\n")) <= 6 for s in secs[:-1])
    assert secs[-1].text.endswith("短尾巴。") and len(secs[-1].text) > 100  # short tail merged
    assert secs[1].prev == secs[0].text


def test_keep_draft_filters_facts_both_ways():
    orig = "2024 年我们用 3 台机器训练，成本 1,200 美元，速度提升 40%。"
    ok = "我们在 2024 年动用了 3 台机器来训练，花了 1200 美元，速度快了 40%。"
    assert orr.keep_draft(orig, ok) == (True, "ok")
    assert orr.keep_draft(orig, ok.replace("40%", "40%，节省了 15 天"))[1] == "number_invented"
    assert orr.keep_draft(orig, "我们动用了几台机器来训练，花了一些钱，速度快了不少，大家都很满意这个结果。")[1] == "numbers_missing"
    # even one missing number is rejected by default: the model would learn to invent it
    assert orr.keep_draft(orig, "我们在 2024 年动用了 3 台机器来训练，花了不少钱，速度快了 40%，很划算。")[1] == "numbers_missing"
    listed = "PhD的优势有: 1) 思维缜密，遇事冷静；2) 技术上眼界广，把层数从10层推到了1000层。"
    prose = "读博带来的好处，一是想问题缜密、遇事冷静，二是技术眼界广，把网络从10层一路推到了1000层。"
    assert orr.keep_draft(listed, prose) == (True, "ok")
    assert orr.keep_draft(listed, prose.replace("从10层一路推到了1000层", "推得很深"))[1] == "numbers_missing"


def test_command_client_runs_in_its_own_dir(tmp_path):
    script = "import pathlib,sys; p=pathlib.Path(sys.argv[1]); pathlib.Path(sys.argv[2]).write_text('【改写后】' + p.read_text()[:4])"
    client = CommandClient({"command": [sys.executable, "-c", script, "{prompt_file}", "{output_file}"], "concurrency": 2}, tmp_path)
    res = asyncio.run(client.complete(None, [{"role": "user", "content": "你好世界！"}], job_id="a#s00_draft"))
    assert res["text"] == "你好世界"
    assert len(list(tmp_path.iterdir())) == 1


def test_multiple_drafters_get_suffixed_prompt_ids(tmp_path):
    from voice_lora.config import Config

    cfg = Config(raw={"workdir": str(tmp_path / "w"), "drafters": [
        {"name": "a", "base_url": "http://localhost:1/v1", "model": "m1", "prompts": ["draft_default", "draft_style"]},
        {"name": "b", "base_url": "http://localhost:2/v1", "model": "m2", "prompts": ["draft_formal"]},
    ]}, base_dir=tmp_path)
    units = [{"unit_id": "p#s00", "text": "原文", "prev": ""}]
    jobs = orr.make_jobs(cfg, units, {"p#s00|draft_default"}, stage="draft", outlines={"p#s00": "要点：1. x"})
    assert [(j["prompt_id"], j["drafter"]) for j in jobs] == [("draft_style", "a"), ("draft_formal@b", "b")]
    assert orr.base_prompt("draft_formal@b") == "draft_formal"


def test_chat_client_reports_provider_reason():
    import httpx
    import pytest

    from voice_lora.llm import ChatClient

    def handler(request):
        return httpx.Response(200, json={"choices": [{"finish_reason": "content_filter"}]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await ChatClient({"base_url": "http://x/v1", "model": "m"}).complete(http, [{"role": "user", "content": "hi"}], retries=1)

    with pytest.raises(RuntimeError, match="content_filter"):
        asyncio.run(run())
