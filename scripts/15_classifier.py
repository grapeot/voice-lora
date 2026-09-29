#!/usr/bin/env python3
"""The voice classifier on its own: check how far it can be trusted, or score any markdown articles.

  check                 train on the train split (as 09_evaluate does), then report on the held-out splits how
                        well it separates the author's paragraphs from each rewriter's, by paragraph length too.
  score FILE [FILE...]  P(author) of each article: length-weighted mean over its prose paragraphs
                        (headings, quotes, tables and code are skipped), plus AI markers per 1,000 characters.

The classifier only sees wording (character n-grams). It cannot see invented facts, dropped arguments or
structure; see docs/classifier.md before reading much into a single score.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import _bootstrap  # noqa: F401
from voice_lora import jsonl
from voice_lora.cli import config_from, parser
from voice_lora.mdproc import split_blocks
from voice_lora.metrics import VoiceClassifier, ai_marker_lexicon, keep_rewrite, marker_rate, train_material

LENGTH_BUCKETS = [(0, 80), (80, 200), (200, 10**9)]
MIN_PARAGRAPH = 20   # shorter prose blocks ("原稿：", captions) carry almost no signal


def fit(cfg):
    units = jsonl.read(cfg.work("units.jsonl"))
    rewrites = jsonl.read(cfg.work("rewrites.jsonl"))
    human, ai = train_material(units, rewrites)
    return units, rewrites, VoiceClassifier(human, ai), ai_marker_lexicon(human, ai)


def separation(clf, human: list[str], ai: list[str]) -> dict:
    ph, pa = clf.p_human(human), clf.p_human(ai)
    y = np.r_[np.ones(len(ph)), np.zeros(len(pa))]
    return {"n_author": len(ph), "n_ai": len(pa), "auc": float(roc_auc_score(y, np.r_[ph, pa])),
            "author_mean": float(ph.mean()), "ai_mean": float(pa.mean()),
            "author_above_half": float((ph > 0.5).mean()), "ai_below_half": float((pa < 0.5).mean())}


def check(cfg) -> None:
    units, rewrites, clf, _ = fit(cfg)
    held = {u["unit_id"]: u for u in units if u["split"] in ("val", "test")}
    human = [u["text"] for u in held.values()]
    by_source: dict[str, list[str]] = defaultdict(list)
    pairs = []  # (original, rewrite) for the length breakdown
    seen = set()
    for r in rewrites:
        u = held.get(r["unit_id"])
        key = f"{r['unit_id']}|{r['prompt_id']}"
        if not u or key in seen or not r.get("rewrite") or not keep_rewrite(u["text"], r["rewrite"])[0]:
            continue
        seen.add(key)
        source = r["prompt_id"].split("@", 1)[1] if "@" in r["prompt_id"] else "first"
        by_source[source].append(r["rewrite"])
        pairs.append((u["text"], r["rewrite"]))
    rows = {"all": separation(clf, human, [t for ts in by_source.values() for t in ts])}
    rows.update({f"source={s}": separation(clf, human, ts) for s, ts in sorted(by_source.items())})
    for lo, hi in LENGTH_BUCKETS:
        h = [t for t in human if lo <= len(t) < hi]
        a = [r for o, r in pairs if lo <= len(o) < hi]
        if h and a:
            rows[f"len={lo}-{hi if hi < 10**9 else ''}"] = separation(clf, h, a)
    print("| 切片 | 原文段 | AI 段 | AUC | 原文均值 | AI 均值 | 原文 >0.5 | AI <0.5 |\n|---|---|---|---|---|---|---|---|")
    for k, m in rows.items():
        print(f"| {k} | {m['n_author']} | {m['n_ai']} | {m['auc']:.3f} | {m['author_mean']:.3f} | {m['ai_mean']:.3f} "
              f"| {m['author_above_half']:.0%} | {m['ai_below_half']:.0%} |")
    cfg.work("eval", "x").parent.mkdir(parents=True, exist_ok=True)
    cfg.work("eval", "classifier_check.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))


def paragraphs(md: str) -> list[str]:
    md = re.sub(r"\A---\n.*?\n---\n", "", md, flags=re.S)  # front matter
    return [b.plain.strip() for b in split_blocks(md) if b.rewrite and len(b.plain.strip()) >= MIN_PARAGRAPH]


def score(cfg, files: list[str], as_json: bool) -> None:
    _, _, clf, lexicon = fit(cfg)
    rows = []
    for f in files:
        paras = paragraphs(Path(f).read_text())
        if not paras:
            continue
        p = clf.p_human(paras)
        w = np.array([len(t) for t in paras], dtype=float)
        rows.append({"file": f, "p_author": float((p * w).sum() / w.sum()), "marker_rate": marker_rate(paras, lexicon),
                     "paragraphs": len(paras), "chars": int(w.sum())})
    rows.sort(key=lambda r: r["p_author"])
    if as_json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    print("| P(作者) | AI 词/千字 | 段数 | 文件 |\n|---|---|---|---|")
    for r in rows:
        print(f"| {r['p_author']:.3f} | {r['marker_rate']:.2f} | {r['paragraphs']} | {r['file']} |")


def main() -> None:
    ap = parser(__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    s = sub.add_parser("score")
    s.add_argument("files", nargs="+")
    s.add_argument("--json", action="store_true")
    args = ap.parse_args()
    cfg = config_from(args)
    if args.cmd == "check":
        check(cfg)
    else:
        score(cfg, args.files, args.json)


if __name__ == "__main__":
    main()
