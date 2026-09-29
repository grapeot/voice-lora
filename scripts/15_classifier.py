#!/usr/bin/env python3
"""The voice classifier on its own (same as `voice-lora check` / `voice-lora score --config`):

  check                 train on the train split (as 09_evaluate does), then report on the held-out splits how
                        well it separates the author's paragraphs from each rewriter's, by paragraph length too.
  score FILE [FILE...]  P(author) of each article: length-weighted mean over its prose paragraphs
                        (headings, quotes, tables and code are skipped), plus AI markers per 1,000 characters.

The classifier only sees wording (character n-grams); see docs/classifier.md before reading much into a score.
"""
import json

import _bootstrap  # noqa: F401
from voice_lora import classify
from voice_lora.cli import config_from, parser


def main() -> None:
    ap = parser(__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    s = sub.add_parser("score")
    s.add_argument("files", nargs="+")
    s.add_argument("--json", action="store_true")
    args = ap.parse_args()
    cfg = config_from(args)
    det, units, rewrites = classify.fit(cfg)
    if args.cmd == "check":
        rows = classify.check(det.clf, units, rewrites)
        print(classify.check_table(rows))
        cfg.work("eval", "classifier_check.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        rows = classify.score_files(det, args.files)
        print(json.dumps(rows, ensure_ascii=False, indent=2) if args.json else classify.score_table(rows))


if __name__ == "__main__":
    main()
