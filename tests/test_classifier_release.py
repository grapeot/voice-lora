import json

import pytest

from voice_lora import classify
from voice_lora.main import main
from voice_lora.metrics import VoiceClassifier

HUMAN = ["我觉得这个事情挺简单的，其实就是多试几次。", "说白了就是钱的问题，没啥好纠结的。"] * 5
AI = ["鉴于上述情况，我们需要借助更多的资源来应对挑战。", "此类问题具有显著的复杂性，需要系统性地开展工作。"] * 5
AI_TEXT = ("鉴于当前行业面临的多重挑战，企业需要借助系统性的方法论来应对复杂局面。具体而言，此类问题往往具有显著的跨部门特征，"
           "唯有建立统一的协调机制，方能确保各项举措落到实处，从而实现降本增效的战略目标。")
HUMAN_TEXT = ("说实话我一开始也没想明白这事。后来跟几个朋友聊了一圈，发现大家踩的坑都差不多：东西买回来，用两次就吃灰了。"
              "所以我现在的做法挺简单的，先借一个用一周，真离不开了再买。")


def test_bands_and_parsing():
    assert [classify.band(p) for p in (0.1, 0.3, 0.49, 0.6, 0.9)] == ["AI 味浓", "灰区", "灰区", "像真人写的", "接近作者本人"]
    custom = classify.parse_bands("0.2,0.4,0.8")
    assert classify.band(0.25, custom) == "灰区" and classify.band(0.85, custom) == "接近作者本人"
    for bad in ("0.5,0.3,0.8", "0.3,0.5"):
        with pytest.raises(ValueError):
            classify.parse_bands(bad)


def test_prune_keeps_the_strongest_features():
    det = classify.Detector(VoiceClassifier(HUMAN, AI), ["鉴于"], "某作者")
    small = det.prune(20)
    assert len(small.clf.vec.vocabulary_) == 20 and small.author == "某作者"
    assert small.clf.p_human([HUMAN[0]])[0] > 0.5 > small.clf.p_human([AI[0]])[0]


def test_bundled_classifier_separates_obvious_cases():
    det = classify.Detector.bundled()
    assert det.author and len(det.clf.vec.vocabulary_) == 10000 and len(det.lexicon) == 300
    p_ai, p_casual = det.clf.p_human([AI_TEXT, HUMAN_TEXT])
    assert p_ai < 0.3 and p_ai < p_casual  # (HUMAN_TEXT was written by a model imitating casual prose: low too, and rightly so)


def test_score_lists_worst_paragraphs_with_markers(tmp_path):
    f = tmp_path / "a.md"
    f.write_text(f"# 标题\n\n{HUMAN_TEXT}\n\n{AI_TEXT}\n\n短句。\n", encoding="utf-8")
    [row] = classify.score_files(classify.Detector.bundled(), [f], worst=1)
    assert row["paragraphs"] == 2 and row["band"] in dict((b, 0) for _, b in classify.BANDS)
    [w] = row["worst"]
    assert w["text"] == AI_TEXT and "鉴于" in w["markers"]


def test_cli_score_uses_bundled_classifier(tmp_path, capsys):
    f = tmp_path / "a.md"
    f.write_text(f"{AI_TEXT}\n", encoding="utf-8")
    main(["score", str(f), "--json", "--paragraphs", "1"])
    [row] = json.loads(capsys.readouterr().out)
    assert row["band"] == "AI 味浓" and row["worst"][0]["markers"]
    main(["score", str(f), "--bands", "0.2,0.4,0.8"])
    assert "| AI 味浓 |" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["score", str(f), "--bands", "0.5"])
