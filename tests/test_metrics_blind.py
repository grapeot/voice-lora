from voice_lora.blind import build_items, score
from voice_lora.evaluation import recommend_checkpoint
from voice_lora.metrics import ai_marker_lexicon, marker_rate, number_errors


def test_number_errors_ignore_formatting_differences():
    assert number_errors(["设定为 100.0 毫米"], ["设定为100毫米"]) == 0
    assert number_errors(["在22年引入"], ["在2022年引入"]) == 0
    assert number_errors(["用了3台"], ["用了三台"]) == 0
    assert number_errors(["提升了40%"], ["提升了50%"]) == 1


def test_marker_lexicon_finds_ai_words():
    human = ["我觉得这个东西挺好用的，就是有点贵。"] * 30
    ai = ["鉴于该产品具备显著优势，笔者认为其极为实用。"] * 30
    lex = ai_marker_lexicon(human, ai, top=20, min_count=5)
    assert any(g in "鉴于该产品具备显著优势" for g in lex)
    assert marker_rate(ai, lex) > marker_rate(human, lex)


def _inputs(n):
    return [{"unit_id": f"u{i}", "ai_text": f"AI{i}", "original": "原" * 100 + str(i)} for i in range(n)]


def test_blind_items_and_scoring():
    inputs = _inputs(70)
    items = build_items(inputs, {r["unit_id"]: "模" + r["unit_id"] for r in inputs}, {r["unit_id"]: "基" + r["unit_id"] for r in inputs}, seed=1)
    assert len(items["s1"]) == 40 and sum(x["mine"] for x in items["s1"]) == 20 and len(items["s2"]) == 20
    fake_ids = {x["unit_id"] for x in items["s1"] if not x["mine"]}
    real_ids = {x["unit_id"] for x in items["s1"] if x["mine"]}
    assert not fake_ids & real_ids  # never the same paragraph on both sides
    answers = {"s1": {x["id"]: True for x in items["s1"]}, "s2": {x["id"]: x["model_side"] for x in items["s2"]}}
    res = score(items, answers)
    assert res["s1_model_judged_author"] == 20 and res["s2_model_wins"] == 20 and res["s2_p_value"] < 0.001


class _FakeScorer:
    def flags(self, m, original):
        return ["overshoot"] if m["p_human"] > original["p_human"] + 0.03 else []


def test_recommendation_skips_overshoot_and_warns_on_rising_loss():
    curve = [(100, 0.25, 1.30), (200, 0.5, 1.29), (400, 1.0, 1.45)]
    m = lambda p: {"p_human": p, "chrf": 30.0, "marker_rate": 1.0, "number_err": 0.0}
    rec = recommend_checkpoint(curve, {100: m(0.85), 200: m(0.95), 400: m(0.97)}, {"p_human": 0.86}, _FakeScorer())
    assert rec["recommended_step"] == 100
    assert any("rose" in n for n in rec["notes"])


def test_grouped_numbers_and_paragraph_guard():
    import asyncio

    from voice_lora.articles import check_paragraph, guarded

    assert number_errors(["相当于 14000 美元"], ["相当于14,000美元"]) == 0
    inp = "普华永道调查了全美 27 家健康保险计划的总精算师，近 70% 的受访者把自动编码工具列为推高成本的前三大原因。"
    assert check_paragraph(inp, "普华永道调查了全美 27 家健康保险的损失甚至都还没有被定义出来。") is not None
    assert check_paragraph(inp, "普华永道问了全美 27 家健康保险计划的总精算师，差不多 70% 的人把自动编码工具排进推高成本的前三。") is None

    calls = []

    async def bad(text, prev):
        calls.append(text)
        return "短"

    safe = guarded(bad, retries=1)
    assert asyncio.run(safe(inp, "")) == inp and len(calls) == 2 and safe.guarded_log
