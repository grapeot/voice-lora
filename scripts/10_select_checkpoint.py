#!/usr/bin/env python3
"""Recommend a checkpoint for a training run from its validation-loss curve and the scored checkpoints.

Score checkpoints first (05_infer_hf.py --name <run>_<step>), then:
  python scripts/10_select_checkpoint.py --config local/config.yaml --run base_05ep

Rule: among checkpoints without flags (overshoot, number_drift), take the lowest validation loss.
Also warns when the loss bottoms out at the first evaluation or climbs after its minimum.
"""
import json

import _bootstrap  # noqa: F401
from voice_lora.cli import config_from, parser
from voice_lora.evaluation import Scorer, recommend_checkpoint, val_loss_curve


def main() -> None:
    ap = parser(__doc__)
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    cfg = config_from(args)
    run_dir = cfg.work("runs", args.run, "x").parent
    curve = val_loss_curve(run_dir)
    scorer = Scorer(cfg)
    original = scorer.reference()["original"]
    scored = {}
    for f in sorted(cfg.work("eval", "x").parent.glob(f"{args.run}_*.jsonl")):
        step = f.stem.rsplit("_", 1)[-1]
        if step.isdigit():
            scored[int(step)] = scorer.score_outputs(scorer.load_system(f))
    print("validation loss:", ", ".join(f"step {s} (ep {e:.2f}) {loss:.4f}" for s, e, loss in curve) or "no log.jsonl")
    if not scored:
        steps = [s for s, _, _ in sorted(curve, key=lambda x: x[2])[:3]]
        print(f"no scored checkpoints yet; score the lowest-loss ones first, e.g. steps {steps}")
        return
    rec = recommend_checkpoint(curve, scored, original, scorer)
    print(f"original: P(作者) {original['p_human']:.3f}, AI 词/千字 {original['marker_rate']:.2f}")
    for c in rec["candidates"]:
        vl = f"{c['val_loss']:.4f}" if c["val_loss"] is not None else "-"
        print(f"  step {c['step']:>5}  val {vl}  P(作者) {c['p_human']:.3f}  AI词 {c['marker_rate']:.2f}  "
              f"chrF {c['chrf']:.1f}  数字错 {c['number_err']:.1%}  {' '.join(c['flags'])}")
    for n in rec["notes"]:
        print("note:", n)
    print("recommended step:", rec["recommended_step"])
    cfg.work("runs", args.run, "selection.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
