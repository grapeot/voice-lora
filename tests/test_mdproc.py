from voice_lora.mdproc import join_blocks, restore_links, split_blocks


def test_markdown_roundtrip_keeps_non_prose_and_links():
    md = "# 标题\n\n第一段，参考[这个项目](https://example.com/a)的做法。\n\n![图](a.png)\n\n```\ncode\n\nmore\n```\n\n| a | b |\n|---|---|\n\n第二段中文。"
    blocks = split_blocks(md)
    assert [b.rewrite for b in blocks] == [False, True, False, False, False, True]
    assert blocks[1].plain == "第一段，参考这个项目的做法。"
    text, lost = restore_links("改写后提到这个项目。", blocks[1].links)
    assert text == "改写后提到[这个项目](https://example.com/a)。" and lost == 0
    for b in blocks:
        if b.rewrite:
            b.output = b.plain
    out = join_blocks(blocks)
    assert "```\ncode\n\nmore\n```" in out and out.startswith("# 标题")


def test_reworded_anchor_is_matched_or_appended():
    links = [("其 2026 财年的实际运营预算只有大约 1500 万美元", "https://example.com/a"), ("完全无关的锚文本内容", "https://example.com/b")]
    text = "它2026财年的实际运营预算大约只有1500万美元，远不够用。"
    out, appended = restore_links(text, links)
    assert "(https://example.com/a)" in out and out.index("[") < out.index("远不够用")
    assert appended == 1 and out.endswith("（[完全无关的锚文本内容](https://example.com/b)）")


def test_fuzzy_link_does_not_cross_punctuation():
    out, _ = restore_links("认为付了这笔钱就覆盖了全部使用成本。SemiAnalysis（芯片与前沿技术研究机构）在2026年做过实测。",
                           [("芯片与前沿技术研究机构 SemiAnalysis", "https://example.com/s")])
    linked = out[out.index("[") + 1 : out.index("](https://example.com/s)")]
    assert "SemiAnalysis" in linked and "。" not in linked


def test_section_rewrite_merges_groups_and_keeps_headings():
    import asyncio
    from pathlib import Path

    from voice_lora.articles import group_blocks, rewrite_article

    md = "# 标题\n\n第一段，见[链接](https://example.com/x)。\n\n第二段内容。\n\n## 小节\n\n第三段内容。"
    blocks = split_blocks(md)
    assert group_blocks(blocks, "section", target_chars=900) == [[1, 2], [4]]
    assert group_blocks(blocks, "paragraph") == [[1], [2], [4]]

    async def fake(text, prev):
        return text.replace("\n\n", "")  # the model may merge paragraphs

    p = Path(__file__).with_name("_tmp_article.md")
    p.write_text(md, encoding="utf-8")
    try:
        out, log = asyncio.run(rewrite_article(p, fake, unit="section"))
    finally:
        p.unlink()
    assert out.startswith("# 标题\n\n第一段，见[链接](https://example.com/x)。第二段内容。\n\n## 小节")
    assert [x["blocks"] for x in log] == [[1, 2], [4]]
