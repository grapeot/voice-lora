import asyncio
import json

import httpx
import pytest

from voice_lora import classify, jsonl
from voice_lora.compare import compare_page, load_log
from voice_lora.main import main
from voice_lora.metrics import VoiceClassifier
from voice_lora.rewrite import ModelCard, request_body, rewrite_files, sampling_overrides

CARD = ModelCard(name="m1", base_url="http://test/v1", model="served-id", instruction="改成作者的文风。")
ARTICLE = "# 标题\n\n第一段讲了[一个项目](https://example.com/a)的经过，花了 42 天。\n\n![图](a.png)\n\n第二段是结论，没有数字。\n"
TOKEN = "【要改写的段落】\n"


def fake_server(bump_numbers: bool = False) -> tuple[httpx.MockTransport, list[dict]]:
    """Echo the paragraph back with a marker; optionally change 42 to 43 so the guard kicks in."""
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        text = body["messages"][0]["content"].split(TOKEN, 1)[1]
        out = text.replace("经过", "来龙去脉").replace("结论", "最后的结论")
        if bump_numbers:
            out = out.replace("42", "43")
        return httpx.Response(200, json={"choices": [{"message": {"content": out}}], "usage": {"completion_tokens": 7}})

    return httpx.MockTransport(handler), seen


def test_request_body_sends_both_penalty_keys_and_defaults():
    b = request_body(CARD, "一段话", "上一段")
    assert b["model"] == "served-id" and b["temperature"] == 0.7 and b["top_p"] == 0.95
    assert b["repetition_penalty"] == b["repeat_penalty"] == 1.05  # LM Studio only reads repeat_penalty
    assert "【上文，仅供参考，不要改写】\n上一段" in b["messages"][0]["content"]
    g = request_body(CARD, "一段话", sampling=sampling_overrides(temperature=0, repetition_penalty=1.0))
    assert g["temperature"] == 0 and g["top_p"] == 1.0 and g["repeat_penalty"] == 1.0


def test_card_roundtrip_and_validation(tmp_path):
    p = tmp_path / "card.yaml"
    p.write_text(CARD.to_yaml(), encoding="utf-8")
    assert ModelCard.load(p) == CARD
    p.write_text("model: x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="base_url"):
        ModelCard.load(p)


def test_rewrite_files_keeps_structure_links_and_writes_log(tmp_path):
    src = tmp_path / "a.md"
    src.write_text(ARTICLE, encoding="utf-8")
    transport, seen = fake_server()
    [res] = asyncio.run(rewrite_files(CARD, [(src, tmp_path / "out" / "a.m1.md")], transport=transport))
    out = (tmp_path / "out" / "a.m1.md").read_text(encoding="utf-8")
    assert out.startswith("# 标题") and "![图](a.png)" in out
    assert "[一个项目](https://example.com/a)的来龙去脉" in out and "最后的结论" in out
    assert res["units"] == 2 and res["kept_original"] == [] and res["tokens"] == 14
    assert len(load_log(tmp_path / "out" / "a.m1.md")) == 2 and all(b["repeat_penalty"] == 1.05 for b in seen)


def test_guard_falls_back_when_numbers_change(tmp_path):
    src = tmp_path / "a.md"
    src.write_text(ARTICLE, encoding="utf-8")
    transport, seen = fake_server(bump_numbers=True)
    [res] = asyncio.run(rewrite_files(CARD, [(src, tmp_path / "a.out.md")], transport=transport))
    assert len(seen) == 3  # the paragraph with 42 is retried once, the other one passes
    assert [k["input"][:3] for k in res["kept_original"]] == ["第一段"]
    assert "花了 42 天" in (tmp_path / "a.out.md").read_text(encoding="utf-8")


HUMAN = ["我觉得这个事情挺简单的，其实就是多试几次。", "说白了就是钱的问题，没啥好纠结的。"] * 5
AI = ["鉴于上述情况，我们需要借助更多的资源来应对挑战。", "此类问题具有显著的复杂性，需要系统性地开展工作。"] * 5


def test_detector_json_roundtrip(tmp_path):
    det = classify.Detector(VoiceClassifier(HUMAN, AI), ["鉴于", "借助"], "某作者")
    det.save(tmp_path / "clf.json")
    back = classify.Detector.load(tmp_path / "clf.json")
    texts = ["其实就是多试几次", "借助资源应对挑战", "无关的一句话"]
    assert back.clf.p_human(texts).tolist() == pytest.approx(det.clf.p_human(texts).tolist())
    assert back.author == "某作者" and back.lexicon == ["鉴于", "借助"]
    (tmp_path / "bad.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        classify.Detector.load(tmp_path / "bad.json")


def test_compare_page_columns_and_scores(tmp_path):
    src = tmp_path / "a.md"
    src.write_text(ARTICLE, encoding="utf-8")
    transport, _ = fake_server()
    asyncio.run(rewrite_files(CARD, [(src, tmp_path / "a.m1.md"), (src, tmp_path / "a.m2.md")], transport=transport))
    cols = [("m1", load_log(tmp_path / "a.m1.md")), ("m2", load_log(tmp_path / "a.m2.md"))]
    page, stats = compare_page(ARTICLE, cols)
    assert "--n:3" in page and "voice-lora · m1 / m2" in page and "像作者写的概率" not in page
    assert stats["links"] == ["1/1", "1/1"]
    det = classify.Detector(VoiceClassifier(HUMAN, AI), ["结论"], "某作者")
    page, stats = compare_page(ARTICLE, cols[:1], det)
    assert "像某作者写的概率" in page and '<mark class="ai">结论</mark>' in page and 0 <= stats["m1"]["p_out"] <= 1


def test_cli_score_and_compare(tmp_path, capsys):
    classify.Detector(VoiceClassifier(HUMAN, AI), ["鉴于"], "某作者").save(tmp_path / "clf.json")
    a = tmp_path / "human.md"
    a.write_text("# t\n\n" + HUMAN[0] * 3 + "\n", encoding="utf-8")
    b = tmp_path / "ai.md"
    b.write_text("# t\n\n" + AI[0] * 3 + "\n", encoding="utf-8")
    main(["score", str(a), str(b), "--classifier", str(tmp_path / "clf.json"), "--json"])
    rows = json.loads(capsys.readouterr().out)
    assert [r["file"] for r in rows] == [str(b), str(a)]  # most AI-like first
    src = tmp_path / "a.md"
    src.write_text(ARTICLE, encoding="utf-8")
    log = [{"input": "x", "output": "第一段改写。", "blocks": [1]}, {"input": "y", "output": "第二段改写。", "blocks": [3]}]
    jsonl.write(tmp_path / "a.voice-v1.1.blocks.jsonl", log)
    main(["compare", str(src), str(tmp_path / "a.voice-v1.1.md"), "--classifier", str(tmp_path / "clf.json")])
    out = json.loads(capsys.readouterr().out)["out"]
    page = (tmp_path / "a.voice-v1.1.compare.html").read_text(encoding="utf-8")
    assert out.endswith("a.voice-v1.1.compare.html") and "第二段改写。" in page and "voice-lora · voice-v1.1" in page
