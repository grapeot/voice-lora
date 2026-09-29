#!/usr/bin/env python3
"""Select the author's posts, cut them into paragraph units and split by post.

Writes <workdir>/units.jsonl, posts.csv and corpus_report.json.
"""
import csv
import json
from collections import Counter
from dataclasses import asdict

import _bootstrap  # noqa: F401
from voice_lora import jsonl
from voice_lora.cli import config_from, parser
from voice_lora.builders import get_builder
from voice_lora.corpus import assign_splits, load_posts, make_units


def main() -> None:
    cfg = config_from(parser(__doc__).parse_args())
    rules = cfg["corpus"]
    paths = cfg.glob(rules["paths"])
    if not paths:
        raise SystemExit(f"no files match corpus.paths {rules['paths']}")
    posts, excluded = load_posts(paths, rules)
    sp = rules.get("split", {})
    splits = assign_splits(posts, sp.get("test_posts", 20), sp.get("val_posts", 10), sp.get("seed", 0), sp.get("min_test_per_bucket"))
    if sp.get("from_file"):
        # Keep an earlier round's split so results stay comparable across rounds; new posts go to train.
        with cfg.path(sp["from_file"]).open() as f:
            fixed = {row["slug"]: row["split"] for row in csv.DictReader(f)}
        splits = {slug: fixed.get(slug, "train") for slug in splits}
    u = rules.get("units", {})
    builder = get_builder(cfg.get("builder.name", "paragraph_rewrite"), cfg.get("builder.allow_experimental", False))
    # Builders may define their own unit (outline_regen cuts sections); the default is paragraph units.
    if hasattr(builder, "make_units"):
        cut = lambda p: builder.make_units(p, cfg)  # noqa: E731
    else:
        cut = lambda p: make_units(p, u.get("min_chars", 50), u.get("max_chars", 800))  # noqa: E731
    units = []
    with cfg.work("posts.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slug", "date", "split", "title", "units", "chars"])
        for p in posts:
            us = cut(p)
            for x in us:
                x.split = splits[p.slug]
            units.extend(us)
            w.writerow([p.slug, p.date, splits[p.slug], p.title, len(us), sum(len(x.text) for x in us)])
    jsonl.write(cfg.work("units.jsonl"), [asdict(x) for x in units])
    report = {
        "files": len(paths),
        "posts": len(posts),
        "excluded": dict(Counter(excluded.values())),
        "units": len(units),
        "chars": sum(len(x.text) for x in units),
        "splits": {s: {"posts": len({x.slug for x in units if x.split == s}), "units": sum(x.split == s for x in units)} for s in ("train", "val", "test")},
    }
    cfg.work("corpus_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
