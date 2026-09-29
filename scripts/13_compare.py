#!/usr/bin/env python3
"""Side-by-side page for one rewritten article: original on the left, rewrite on the right, paragraph by
paragraph, with AI-marker words underlined on both sides and before/after scores on top.

  python scripts/13_compare.py --config local/config.yaml --article path/to/article.md --name lmstudio_v1

Reads <workdir>/eval/articles/<stem>.<name>.md (+ .blocks.jsonl, written by 08_rewrite.py) and writes
<workdir>/eval/articles/<stem>.<name>.compare.html (a body-only page: open it locally or publish it).
"""
import html
import json
import re

import _bootstrap  # noqa: F401
from voice_lora import jsonl
from voice_lora.cli import config_from, parser
from voice_lora.evaluation import Scorer
from voice_lora.mdproc import LINK_RE, restore_links, split_blocks
from voice_lora.metrics import marker_rate

STYLE = """
:root { --paper:#f3f4ef; --sheet:#fbfbf8; --ink:#1d2320; --muted:#5d665f; --rule:#d9ddd3; --grid:#b9cdb9;
  --red:#b3322a; --indigo:#2d4876; --indigo-soft:rgba(45,72,118,.08);
  --serif:"Noto Serif SC","Songti SC","STSong",serif; --sans:"Noto Sans SC","PingFang SC","Hiragino Sans GB",system-ui,sans-serif; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme:dark; --paper:#131715; --sheet:#1a1f1c; --ink:#e3e7e1;
  --muted:#9aa39b; --rule:#2c332e; --grid:#33483a; --red:#e46a5c; --indigo:#97b1e0; --indigo-soft:rgba(151,177,224,.10); } }
:root[data-theme="dark"] { color-scheme:dark; --paper:#131715; --sheet:#1a1f1c; --ink:#e3e7e1; --muted:#9aa39b; --rule:#2c332e;
  --grid:#33483a; --red:#e46a5c; --indigo:#97b1e0; --indigo-soft:rgba(151,177,224,.10); }
* { box-sizing:border-box; }
body { background:var(--paper); color:var(--ink); font-family:var(--sans); font-size:15px; line-height:1.7; margin:0; padding-inline:16px; padding-block:0 56px; }
header { margin-inline:-16px; padding:36px 16px 26px; border-bottom:1px solid var(--rule); background-color:var(--sheet);
  background-image:linear-gradient(to right,var(--grid) 1px,transparent 1px),linear-gradient(to bottom,var(--grid) 1px,transparent 1px); background-size:28px 28px; }
.wrap { max-width:1180px; margin:0 auto; }
.card { background:var(--sheet); border:1px solid var(--rule); padding:20px 22px; max-width:820px; display:grid; gap:10px; }
.eyebrow { font-size:12px; letter-spacing:.1em; color:var(--muted); }
h1 { font-family:var(--serif); font-weight:900; font-size:clamp(24px,4vw,36px); line-height:1.25; margin:0; text-wrap:balance; }
p { margin:0; max-width:70ch; }
.stats { display:flex; gap:22px; flex-wrap:wrap; font-size:13.5px; color:var(--muted); }
.stats b { color:var(--ink); font-variant-numeric:tabular-nums; }
.cols { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:28px; }
.head { position:sticky; top:env(safe-area-inset-top,0px); background:var(--paper); border-bottom:1px solid var(--rule); padding-block:10px; z-index:2; font-size:13px; color:var(--muted); margin-top:28px; }
.head b { color:var(--ink); display:block; font-size:14px; }
.row { padding-block:14px; border-bottom:1px dashed var(--rule); }
.prose { font-family:var(--serif); font-size:16px; line-height:1.9; white-space:pre-wrap; }
.right { background:var(--indigo-soft); margin:-6px; padding:6px; }
.sec { font-family:var(--serif); font-weight:600; font-size:20px; padding-top:26px; padding-bottom:4px; }
mark.ai { background:none; color:var(--red); text-decoration:underline wavy var(--red); text-underline-offset:4px; text-decoration-thickness:1px; }
a { color:var(--indigo); }
.label { display:none; font-size:11.5px; color:var(--muted); }
.legend { font-size:13px; color:var(--muted); margin-top:14px; }
@media (max-width:760px) { .cols { grid-template-columns:1fr; gap:8px; } .head { display:none; } .label { display:block; } .right { margin:0; } }
"""


def render(text: str, lex_re) -> str:
    """Escape, underline AI markers, then turn markdown links into anchors."""
    links = []

    def hold(m):
        links.append((m.group(1), m.group(2)))
        return f"\u0000{len(links) - 1}\u0000"

    text = LINK_RE.sub(hold, text)
    out = html.escape(text)
    if lex_re:
        out = lex_re.sub(lambda m: f'<mark class="ai">{m.group(0)}</mark>', out)
    def anchor(m):
        label, url = links[int(m.group(1))]
        return f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener">{html.escape(label)}</a>'

    return re.sub("\u0000(\\d+)\u0000", anchor, out)


def main() -> None:
    ap = parser(__doc__)
    ap.add_argument("--article", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    cfg = config_from(args)
    src = cfg.path(args.article)
    base = cfg.work("eval", "articles", f"{src.stem}.{args.name}.md")
    left_blocks = split_blocks(src.read_text(encoding="utf-8"))
    log = jsonl.read(base.with_suffix(".blocks.jsonl"))
    scorer = Scorer(cfg)
    lex = sorted(scorer.lexicon[:120], key=len, reverse=True)
    lex_re = re.compile("|".join(re.escape(g) for g in lex)) if lex else None
    ins, outs = [x["input"] for x in log], [x["output"] for x in log]
    stats = {"p_in": float(scorer.clf.p_human(ins).mean()), "p_out": float(scorer.clf.p_human(outs).mean()),
             "m_in": marker_rate(ins, scorer.lexicon), "m_out": marker_rate(outs, scorer.lexicon)}
    n_links = sum(len(LINK_RE.findall(b.text)) for b in left_blocks)
    title = args.title or next((b.text.lstrip("# ").strip() for b in left_blocks if b.text.startswith("# ")), src.stem)
    author = cfg.get("author.name", "作者")

    # The paragraph log is in the order of the original's prose blocks; align on it rather than re-parsing
    # the rewritten markdown, whose paragraph count can differ when the model splits a paragraph.
    prose = [b for b in left_blocks if b.rewrite]
    if len(prose) != len(log):
        raise SystemExit(f"{len(prose)} prose paragraphs in the original but {len(log)} in the log; rerun 08_rewrite.py")
    right = {id(b): restore_links(x["output"], b.links)[0] for b, x in zip(prose, log, strict=True)}
    n_links_out = sum(len(LINK_RE.findall(t)) for t in right.values())
    rows = []
    for lb in left_blocks:
        if not lb.rewrite:
            if lb.text.startswith("#") and not lb.text.startswith("# "):
                rows.append(f'<div class="sec">{html.escape(lb.text.lstrip("# ").strip())}</div>')
            continue
        rows.append('<div class="row cols"><div><span class="label">原稿</span><div class="prose">' + render(lb.text, lex_re) +
                    '</div></div><div class="right"><span class="label">改写</span><div class="prose">' + render(right[id(lb)], lex_re) + "</div></div></div>")
    page = f"""<title>{html.escape(title)} 改写对照</title>
<style>{STYLE}</style>
<header><div class="wrap"><div class="card">
<div class="eyebrow">voice-lora · {html.escape(args.name)}</div>
<h1>{html.escape(title)}</h1>
<p>左栏是原稿，右栏是模型改写成{html.escape(author)}文风的版本，逐段对齐；标题、图片和表格原样保留，这里只列正文。</p>
<div class="stats"><span>像{html.escape(author)}写的概率 <b>{stats['p_in']:.2f} → {stats['p_out']:.2f}</b></span>
<span>AI 腔词 / 千字 <b>{stats['m_in']:.1f} → {stats['m_out']:.1f}</b></span>
<span>正文段落 <b>{len(log)}</b></span><span>链接 <b>{n_links_out} / {n_links}</b></span></div>
</div></div></header>
<div class="wrap">
<div class="head cols"><div><b>原稿</b>AI 起草的版本</div><div><b>改写</b>{html.escape(args.name)}</div></div>
{''.join(rows)}
<p class="legend"><mark class="ai">波浪线</mark>：命中 AI 腔词表的词（训练数据里 AI 改写远多于作者原文的词）。</p>
</div>"""
    out = base.with_suffix(".compare.html")
    out.write_text(page, encoding="utf-8")
    print(json.dumps({**stats, "links": f"{n_links_out}/{n_links}", "out": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
