#!/usr/bin/env python3
"""Score every system in <workdir>/eval/*.jsonl on the held-out paragraphs, plus whole-article outputs,
against two references: the AI input (floor) and the author's original (ceiling).

Writes <workdir>/eval/metrics.json and metrics.md. Flags:
  overshoot     more "author-like" than the author (v1 hit this from 1 epoch on); pick an earlier checkpoint
  number_drift  more than 5% of paragraphs change a number
"""
import json

import _bootstrap  # noqa: F401
from voice_lora.cli import config_from, parser
from voice_lora.evaluation import Scorer

COLS = [("p_human", "P(作者)", "{:.3f}"), ("share_human", ">0.5", "{:.0%}"), ("marker_rate", "AI 词/千字", "{:.2f}"),
        ("chrf", "chrF", "{:.1f}"), ("number_err", "数字错", "{:.1%}"), ("len_ratio", "长度比", "{:.2f}")]


def main() -> None:
    cfg = config_from(parser(__doc__).parse_args())
    scorer = Scorer(cfg)
    ref = scorer.reference()
    systems = {"ai_input": ref["ai_input"]}
    for f in sorted(cfg.work("eval", "x").parent.glob("*.jsonl")):
        systems[f.stem] = scorer.score_outputs(scorer.load_system(f))
    systems["original"] = ref["original"]
    flags = {n: scorer.flags(m, ref["original"]) for n, m in systems.items() if n not in ("ai_input", "original")}
    articles = scorer.article_scores()

    lines = ["| 系统 | " + " | ".join(c[1] for c in COLS) + " | 标记 |", "|---" * (len(COLS) + 2) + "|"]
    for n, m in systems.items():
        lines.append(f"| {n} | " + " | ".join(fmt.format(m[k]) for k, _, fmt in COLS) + f" | {' '.join(flags.get(n, []))} |")
    if articles:
        lines += ["", "整篇文章（逐段改写后的段落）：", "", "| 系统 | P(作者) | AI 词/千字 | 段数 |", "|---|---|---|---|"]
        for n, m in articles.items():
            lines.append(f"| {n} | {m['p_human']:.3f} | {m['marker_rate']:.2f} | {m['paragraphs']} |")
    table = "\n".join(lines)
    cfg.work("eval", "metrics.md").write_text(table + "\n")
    cfg.work("eval", "metrics.json").write_text(json.dumps({"paragraphs": systems, "flags": flags, "articles": articles}, ensure_ascii=False, indent=2))
    print(table)


if __name__ == "__main__":
    main()
