#!/usr/bin/env python3
"""Make the blind-test page: a static HTML file the author opens locally. Answers stay in the browser
and are exported as JSON for 12_score_blind.py.

  python scripts/11_blind_eval.py --config local/config.yaml --model base_05ep --baseline baseline_fewshot

Writes <workdir>/blind/<model>/items.json and blind_eval.html.
"""
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from voice_lora import jsonl
from voice_lora.blind import build_items, render_page
from voice_lora.cli import config_from, parser

TEMPLATE = Path(__file__).resolve().parents[1] / "blind_eval" / "template.html"


def main() -> None:
    ap = parser(__doc__)
    ap.add_argument("--model", required=True, help="system name under <workdir>/eval/")
    ap.add_argument("--baseline", default="baseline_fewshot")
    ap.add_argument("--n", type=int, default=20, help="items per task")
    args = ap.parse_args()
    cfg = config_from(args)
    load = lambda name: {r["unit_id"]: r["output"] for r in jsonl.read(cfg.work("eval", f"{name}.jsonl"))}
    items = build_items(jsonl.read(cfg.work("eval_inputs.jsonl")), load(args.model), load(args.baseline), args.n, args.n,
                        seed=int(cfg.get("corpus.split.seed", 0)))
    out_dir = cfg.work("blind", args.model, "items.json").parent
    (out_dir / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=2))
    page = render_page(TEMPLATE.read_text(encoding="utf-8"), items, eval_id=args.model, author=cfg.get("author.name", "作者"),
                       baseline_name=args.baseline)
    (out_dir / "blind_eval.html").write_text(page, encoding="utf-8")
    print("open", out_dir / "blind_eval.html", "in a browser; export the answers when done")


if __name__ == "__main__":
    main()
