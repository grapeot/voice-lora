#!/usr/bin/env python3
"""Score exported blind-test answers against the answer key.

  python scripts/12_score_blind.py --config local/config.yaml --model base_05ep --answers ~/Downloads/blind_answers_base_05ep.json

Writes <workdir>/blind/<model>/score.json. Read the result with its ceiling in mind: authors do not
recognise all of their own paragraphs either (in v1 the author accepted 17 of 20 real ones).
"""
import json

import _bootstrap  # noqa: F401
from voice_lora.blind import load_answers, score
from voice_lora.cli import config_from, parser


def main() -> None:
    ap = parser(__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--answers", required=True)
    args = ap.parse_args()
    cfg = config_from(args)
    d = cfg.work("blind", args.model, "items.json").parent
    items = json.loads((d / "items.json").read_text())
    result = score(items, load_answers(args.answers))
    (d / "score.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
