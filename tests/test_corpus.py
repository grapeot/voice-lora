from datetime import date
from pathlib import Path

from voice_lora.corpus import Post, assign_splits, clean_body, exclusion_reason, make_units, parse_post, split_long

ZH = "这是一段用来测试的中文文字，内容本身没有什么意义，只是为了凑够长度而已。" * 2
RULES = {
    "min_cjk_ratio": 0.15,
    "exclude": {"filename_suffixes": ["-en.md"], "tags": ["English"], "statuses": ["draft"], "date_ranges": [["2020-01-01", "2020-06-30"]]},
}


def write(tmp_path, name, header, body=ZH):
    p = tmp_path / name
    p.write_text(header + "\n\n" + body, encoding="utf-8")
    return p


def reason(p):
    meta, body = parse_post(p)
    return exclusion_reason(p, meta, body, RULES)


def test_date_range_is_inclusive(tmp_path):
    for d, expected in [("2019-12-31", None), ("2020-01-01", "date_range"), ("2020-06-30 23:00", "date_range"), ("2020-07-01", None)]:
        p = write(tmp_path, f"p{d[:10]}.md", f"Title: t\nDate: {d}\nTags: Chinese")
        assert reason(p) == expected, d


def test_english_and_draft_rules(tmp_path):
    assert reason(write(tmp_path, "a-en.md", "Date: 2019-01-01")) == "filename_suffix"
    assert reason(write(tmp_path, "b.md", "Date: 2019-01-01\nTags: English, AI")) == "tag"
    assert reason(write(tmp_path, "c.md", "Date: 2019-01-01\nStatus: draft")) == "status"
    english = "This post is written entirely in English with no Chinese at all. " * 5
    assert reason(write(tmp_path, "d.md", "Date: 2019-01-01\nTags: Chinese", english)) == "not_chinese"
    assert reason(write(tmp_path, "e.md", "Title: no date")) == "no_date"


def test_unclosed_fenced_frontmatter(tmp_path):
    p = write(tmp_path, "f.md", "---\nTitle: t\nDate: 2024-03-09 17:00\nTags: Chinese")
    meta, body = parse_post(p)
    assert meta["Date"].startswith("2024-03-09")
    assert ZH in body


def test_clean_body_keeps_anchor_text_and_drops_noise():
    body = ("# 标题\n\n看[这篇文章](https://example.com)就懂了。\n\n![图](a.png)\n\n```python\nprint(1)\n```\n\n> 别人说的话\n\n"
            '<script async src="https://example.com/x.js"></script>')
    out = clean_body(body)
    assert "看这篇文章就懂了。" in out
    for noise in ("标题", "https://", "a.png", "print", "别人说的话", "x.js"):
        assert noise not in out


def test_split_long_cuts_at_sentence_end():
    text = "一二三四五六七八九十。" * 100
    parts = split_long(text, 300)
    assert all(len(p) <= 300 for p in parts) and "".join(parts) == text and all(p.endswith("。") for p in parts)


def test_units_merge_short_and_carry_prev():
    post = Post(Path("x.md"), "x", "t", date(2020, 1, 1), [], "短句。\n\n" + ZH + "\n\n" + ZH)
    units = make_units(post)
    assert len(units) == 2 and units[0].text.startswith("短句。")
    assert units[1].prev == units[0].text and units[0].prev == ""


def test_splits_are_per_post_and_respect_bucket_minimum():
    posts = [Post(Path(f"{i}.md"), f"s{i}", "", date(2012 + i % 15, 6, 1), [], "") for i in range(150)]
    splits = assign_splits(posts, 20, 10, seed=1, min_test_per_bucket={"2026": 3})
    assert set(splits) == {p.slug for p in posts}
    assert sum(v == "test" for v in splits.values()) >= 15
    assert sum(1 for p in posts if p.date.year == 2026 and splits[p.slug] == "test") >= 3
    # every year keeps at least one training post
    for y in range(2012, 2027):
        assert any(splits[p.slug] == "train" for p in posts if p.date.year == y)
