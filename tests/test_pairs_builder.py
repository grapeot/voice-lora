import pytest

from voice_lora.builders import get_builder
from voice_lora.builders import paragraph_rewrite as pr
from voice_lora.config import Config
from voice_lora.pairs import keep_rewrite, to_example, user_message

ORIG = "我在2024年用了3台5090训练模型，速度提升了40%，Unsloth很好用。"


def cfg(tmp_path, **extra):
    raw = {"author": {"name": "测试作者"}, "workdir": str(tmp_path / "w"), "corpus": {"split": {"seed": 3}},
           "builder": {"prompts": list(pr.PROMPTS), "identity_fraction": 0.5}, **extra}
    return Config(raw=raw, base_dir=tmp_path)


def test_keep_rewrite_rules():
    assert keep_rewrite(ORIG, "2024年，我借助三台5090完成了模型训练，速度提升40%，Unsloth的体验相当出色。") == (True, "ok")
    assert keep_rewrite(ORIG, "我在2024年用了3台5090训练模型，速度提升了50%，Unsloth很好用。")[1] == "number_changed"
    assert keep_rewrite(ORIG, "我在2024年用了3台5090训练，速度提升40%，Unsloth和Apple Vision Pro都好用。")[1] == "latin_invented"
    assert keep_rewrite(ORIG, "好。")[1] == "length_ratio"


def test_translating_english_terms_is_allowed():
    o = "这就是prompt engineering的难点，写好prompt很难，需要反复试，还要不断调整。"
    r = "这正是提示词工程的难点所在：撰写优质的提示词并非易事，需要反复尝试与持续调整。"
    assert keep_rewrite(o, r) == (True, "ok")


def test_instruction_comes_from_config(tmp_path):
    c = cfg(tmp_path)
    assert c.instruction == "把下面这段文字改写成测试作者的文风，内容和顺序不变。"
    ex = to_example(c.instruction, {"unit_id": "a#000", "text": "原文", "prev": "上一段"}, "AI 文")
    assert ex["messages"][0]["content"] == user_message(c.instruction, "AI 文", "上一段")
    assert ex["messages"][1]["content"] == "原文"


def test_style_mimic_is_deterministic_and_needs_exemplars():
    unit = {"unit_id": "a#001", "text": "目标段落", "prev": ""}
    pool = [f"示例段落{i}" for i in range(10)]
    m1 = pr.build_messages(unit, "style_mimic", pool)
    assert m1 == pr.build_messages(unit, "style_mimic", pool) and "【示例一】" in m1[0]["content"]
    with pytest.raises(ValueError):
        pr.build_messages(unit, "style_mimic", [])


def test_build_splits_filters_and_picks_eval_input(tmp_path):
    c = cfg(tmp_path)
    units = [{"unit_id": f"p{i}#000", "text": ORIG, "prev": "", "split": s} for i, s in enumerate(["train", "train", "val", "test"])]
    good = "2024年，我借助三台5090完成了模型训练，速度提升40%，Unsloth的体验相当出色。"
    bad = "我在2024年用了3台5090训练模型，速度提升了50%，Unsloth很好用。"
    rewrites = [{"unit_id": u["unit_id"], "prompt_id": p, "rewrite": good} for u in units for p in ("default", "style_mimic")]
    rewrites.append({"unit_id": "p0#000", "prompt_id": "formal", "rewrite": bad})
    examples, eval_inputs, reasons = pr.build(c, units, rewrites)
    assert reasons["formal:number_changed"] == 1
    assert len([e for e in examples["train"] if e["messages"][0]["content"].endswith(good)]) == 4
    assert eval_inputs == [{"unit_id": "p3#000", "prompt_id": "style_mimic", "ai_text": good, "prev": "", "original": ORIG}]


def test_unknown_prompt_and_builder_are_rejected(tmp_path):
    c = cfg(tmp_path, builder={"prompts": ["nope"]})
    with pytest.raises(ValueError):
        pr.make_jobs(c, [{"unit_id": "x", "text": "t"}], set())
    with pytest.raises(ValueError):
        get_builder("nope")
    with pytest.raises(ValueError):  # draft stage needs outlines first
        get_builder("outline_regen").make_jobs(c, [{"unit_id": "x", "text": "t"}], set(), stage="draft", outlines={})
