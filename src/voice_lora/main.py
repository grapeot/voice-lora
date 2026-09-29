"""`voice-lora`: use a trained voice model and its classifier without an experiment workdir.

  voice-lora rewrite article.md --card model.yaml [--out out.md | --out-dir DIR]
  voice-lora compare article.md out.md [out2.md ...] [--classifier clf.json] [--out page.html]
  voice-lora score a.md b.md ... --classifier clf.json [--json]
  voice-lora card --config local/config.yaml --model <served id> --base-url http://localhost:1234/v1 [--out card.yaml]
  voice-lora fit --config local/config.yaml --out clf.json
  voice-lora check --config local/config.yaml

A model card (see card.example.yaml) says where a model is served and which instruction it was trained with;
`card` writes one from the training config so the two cannot drift apart. The training pipeline itself stays in
scripts/01..14 (experiment runs and GPU-only steps).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from . import classify
from .config import load_config
from .rewrite import ModelCard, rewrite_files, sampling_overrides


def _card(args: argparse.Namespace) -> ModelCard:
    if args.card:
        card = ModelCard.load(args.card)
        if args.serve_url:
            card.base_url = args.serve_url
        if args.serve_model:
            card.model = args.serve_model
        return card
    if not args.config:
        raise SystemExit("give --card (a model card) or --config (the training config)")
    return ModelCard.from_config(load_config(args.config), args.serve_url, args.serve_model)


def _detector(args: argparse.Namespace, required: bool = True) -> classify.Detector | None:
    if getattr(args, "classifier", None):
        return classify.Detector.load(args.classifier)
    if getattr(args, "config", None):
        return classify.fit(load_config(args.config))[0]
    if required:
        raise SystemExit("give --classifier (a saved classifier) or --config (fit one from a run)")
    return None


def cmd_rewrite(args: argparse.Namespace) -> None:
    card = _card(args)
    sources = [Path(a).expanduser().resolve() for a in args.articles]
    if args.out and len(sources) > 1:
        raise SystemExit("--out takes one article; use --out-dir for several")
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else Path.cwd()
    jobs = [(src, Path(args.out).expanduser() if args.out else out_dir / f"{src.stem}.{card.name}.md") for src in sources]
    results = asyncio.run(rewrite_files(card, jobs, args.concurrency, args.unit, args.mode, not args.no_guard,
                                        sampling_overrides(args.temperature, args.repetition_penalty)))
    for r in results:
        print(json.dumps(r, ensure_ascii=False), flush=True)


def cmd_compare(args: argparse.Namespace) -> None:
    from .compare import compare_page, load_log

    original = Path(args.original).expanduser()
    # <original stem>.<model name>.md, as `rewrite` names them; model names may contain dots (voice-lora-v1.1).
    names = args.names or [Path(p).stem.removeprefix(original.stem + ".") for p in args.rewritten]
    if len(names) != len(args.rewritten):
        raise SystemExit("--names needs one name per rewritten file")
    det = _detector(args, required=False)
    columns = [(n, load_log(p)) for n, p in zip(names, args.rewritten, strict=True)]
    try:
        page, stats = compare_page(original.read_text(encoding="utf-8"), columns, det, args.title,
                                   args.author, original.stem)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    out = Path(args.out).expanduser() if args.out else Path(args.rewritten[0]).expanduser().with_suffix(".compare.html")
    out.write_text(page, encoding="utf-8")
    print(json.dumps({**stats, "out": str(out)}, ensure_ascii=False))


def cmd_score(args: argparse.Namespace) -> None:
    rows = classify.score_files(_detector(args), args.files)
    print(json.dumps(rows, ensure_ascii=False, indent=2) if args.json else classify.score_table(rows))


def cmd_check(args: argparse.Namespace) -> None:
    det, units, rewrites = classify.fit(load_config(args.config))
    print(classify.check_table(classify.check(det.clf, units, rewrites)))


def cmd_fit(args: argparse.Namespace) -> None:
    det, _, _ = classify.fit(load_config(args.config))
    det.save(args.out)
    print(json.dumps({"out": args.out, "features": len(det.clf.vec.vocabulary_), "lexicon": len(det.lexicon)}, ensure_ascii=False))


def cmd_card(args: argparse.Namespace) -> None:
    card = ModelCard.from_config(load_config(args.config), args.base_url, args.model, args.name)
    if args.out:
        Path(args.out).expanduser().write_text(card.to_yaml(), encoding="utf-8")
        print(args.out)
    else:
        sys.stdout.write(card.to_yaml())


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="voice-lora", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("rewrite", help="rewrite markdown articles with a served voice model")
    p.add_argument("articles", nargs="+")
    p.add_argument("--card", help="model card (YAML): base_url, model, instruction")
    p.add_argument("--config", help="training config, instead of a card")
    p.add_argument("--serve-url", help="override the card's base_url")
    p.add_argument("--serve-model", help="override the card's model")
    p.add_argument("--out", help="output file (one article only); default <out-dir>/<stem>.<card name>.md")
    p.add_argument("--out-dir", help="output directory (default: current directory)")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--unit", choices=["paragraph", "section"], default="paragraph")
    p.add_argument("--mode", choices=["parallel", "chained"], default="parallel")
    p.add_argument("--no-guard", action="store_true", help="keep unsafe rewrites instead of retrying / falling back to the input")
    p.add_argument("--temperature", type=float, help="default 0.7; 0 = greedy (mostly copies; only for checking deployments)")
    p.add_argument("--repetition-penalty", type=float, help="default 1.05; 1.0 turns it off")
    p.set_defaults(func=cmd_rewrite)

    p = sub.add_parser("compare", help="side-by-side HTML page: original vs one or more rewrites")
    p.add_argument("original")
    p.add_argument("rewritten", nargs="+", help="rewritten .md files (their .blocks.jsonl logs must sit next to them)")
    p.add_argument("--names", nargs="+", help="column names (default: the last suffix of each file name)")
    p.add_argument("--classifier", help="saved classifier (JSON) for scores and AI-marker highlights")
    p.add_argument("--config", help="fit the classifier from this run instead")
    p.add_argument("--author", help="author name on the page (default: from the classifier)")
    p.add_argument("--title")
    p.add_argument("--out", help="default: <first rewritten>.compare.html")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("score", help="P(author) and AI markers per article")
    p.add_argument("files", nargs="+")
    p.add_argument("--classifier", help="saved classifier (JSON)")
    p.add_argument("--config", help="fit the classifier from this run instead")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("check", help="how well the classifier separates the author from each rewriter (held-out splits)")
    p.add_argument("--config", required=True)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("fit", help="fit the classifier from a run and save it as JSON")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("card", help="write a model card from the training config")
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True, help="the model's identifier on the server")
    p.add_argument("--base-url", help="default: serve.base_url from the config")
    p.add_argument("--name", help="short name used in output file names (default: the model id)")
    p.add_argument("--out")
    p.set_defaults(func=cmd_card)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
