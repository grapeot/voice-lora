#!/usr/bin/env python3
"""Side-by-side page for one rewritten article: original on the left, one column per rewrite on the right,
paragraph by paragraph, with AI-marker words underlined and before/after scores on top.
Outside an experiment workdir, `voice-lora compare` does the same for any files.

  python scripts/13_compare.py --config local/config.yaml --article path/to/article.md --name lmstudio_v1
  python scripts/13_compare.py --config local/config.yaml --article path/to/article.md --name v1 v11_210   # 3 columns

Reads <workdir>/eval/articles/<stem>.<name>.md (+ .blocks.jsonl, written by 08_rewrite.py) and writes
<workdir>/eval/articles/<stem>.<name>[+<name>...].compare.html (a body-only page: open it locally or publish it).
Several names must come from the same unit (paragraph or section), so their rows line up.
"""
import json

import _bootstrap  # noqa: F401
from voice_lora import classify
from voice_lora.cli import config_from, parser
from voice_lora.compare import compare_page, load_log


def main() -> None:
    ap = parser(__doc__)
    ap.add_argument("--article", required=True)
    ap.add_argument("--name", required=True, nargs="+")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    cfg = config_from(args)
    src = cfg.path(args.article)
    columns = [(n, load_log(cfg.work("eval", "articles", f"{src.stem}.{n}.md"))) for n in args.name]
    det, _, _ = classify.fit(cfg)
    try:
        page, stats = compare_page(src.read_text(encoding="utf-8"), columns, det, args.title, cfg.get("author.name", "作者"), src.stem)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    out = cfg.work("eval", "articles", f"{src.stem}.{'+'.join(args.name)}.compare.html")
    out.write_text(page, encoding="utf-8")
    print(json.dumps({**stats, "out": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
