import asyncio
import sys
from datetime import date
from pathlib import Path

from voice_lora.builders import outline_regen as orr
from voice_lora.corpus import Post, make_sections
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
