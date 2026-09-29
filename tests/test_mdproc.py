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


def test_quote_right_after_text_passes_through():
    blocks = split_blocks("AI 原稿：\n> 原样保留的引用。\n> 第二行。\n\n正文段落。")
    assert [(b.rewrite, b.text) for b in blocks] == [
        (True, "AI 原稿："), (False, "> 原样保留的引用。\n> 第二行。"), (True, "正文段落。")]


def test_fuzzy_links_are_not_cut_mid_word():
    cases = [  # (anchor, rewritten paragraph, expected linked span): real v1.1 outputs that were cut before
        ("芯片与前沿技术研究机构 SemiAnalysis", "认为付了这笔钱就够了。芯片与前沿技术研究机构SemiAnalysis在2026年6月做过一组实测。",
         "芯片与前沿技术研究机构SemiAnalysis"),
        ("SemiAnalysis 行业调研估算的转述", "没有免费午餐。据 SemiAnalysis 行业调研估算，Anthropic 大约 75% 的收入来自合同。",
         "SemiAnalysis 行业调研估算"),
        ("按完成任务结账", "OpenAI正在对一部分企业大客户试行按完成任务结算的模式，官方没有置评。", "按完成任务结算"),
        ("塔夫茨大学学者 Josephine Wolff", "塔夫茨大学的学者 Josephine Wolff 强调过，历史损失数据不足以建模。", "塔夫茨大学的学者 Josephine Wolff"),
    ]
    for anchor, text, want in cases:
        out, appended = restore_links(text, [(anchor, "https://example.com/x")])
        assert appended == 0 and f"[{want}](https://example.com/x)" in out, out


def test_heading_and_code_fence_split_without_blank_lines():
    md = "## 适合使用\n- 第一条说明文字\n- 第二条说明文字\n\n说明如下：\n```bash\n# 注释\nvoice-lora score a.md\n```\n后面的正文。\n"
    blocks = split_blocks(md)
    assert [(b.text.split("\n")[0], b.rewrite, b.tight) for b in blocks] == [
        ("## 适合使用", False, False), ("- 第一条说明文字", True, True), ("说明如下：", True, False),
        ("```bash", False, True), ("后面的正文。", True, True)]
    for b in blocks:
        if b.rewrite:
            b.output = b.plain
    assert join_blocks(blocks) == md


def test_real_documents_round_trip_unchanged():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for f in [root / "README.md", root / "docs" / "classifier.md", *sorted((root / "skills").glob("*/SKILL.md"))]:
        md = f.read_text(encoding="utf-8").rstrip("\n") + "\n"
        md = "\n".join(line.rstrip() for line in md.split("\n"))
        while "\n\n\n" in md:
            md = md.replace("\n\n\n", "\n\n")
        blocks = split_blocks(md)
        for b in blocks:
            if b.rewrite:
                b.output = b.text
        assert join_blocks(blocks) == md, f.name
