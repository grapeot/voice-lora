"""Score systems on the held-out paragraphs and on whole articles, flag the failure modes seen in v1,
and recommend a checkpoint.

Failure modes encoded here (all observed in v1):
- overshoot: the output scores as *more* "author-like" than the author's own text (classifier probability
  above the originals, AI-marker rate far below them). In v1 this appeared from 1 epoch on, together with
  caricature (crude over-the-top slang) and rising validation loss.
- number drift: the output drops or changes numbers from its input.
- paragraph/article gap: good paragraph scores that do not carry over to real AI-drafted articles.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import jsonl
from .config import Config
from .metrics import VoiceClassifier, ai_marker_lexicon, marker_rate, score_system, train_material

OVERSHOOT_P_MARGIN = 0.03        # p_human above the originals' by more than this
OVERSHOOT_MARKER_RATIO = 0.6     # marker rate below 60% of the originals'
NUMBER_ERR_LIMIT = 0.05          # share of paragraphs with a changed number


class Scorer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        units = jsonl.read(cfg.work("units.jsonl"))
        rewrites = jsonl.read(cfg.work("rewrites.jsonl"))
        human, ai = train_material(units, rewrites)
        self.clf = VoiceClassifier(human, ai)
        self.lexicon = ai_marker_lexicon(human, ai)
        self.inputs = jsonl.read(cfg.work("eval_inputs.jsonl"))
        self.order = [r["unit_id"] for r in self.inputs]

    def score_outputs(self, outputs: list[str]) -> dict:
        return score_system(outputs, [r["ai_text"] for r in self.inputs], [r["original"] for r in self.inputs], self.clf, self.lexicon)

    def load_system(self, path: Path) -> list[str]:
        rows = {r["unit_id"]: r["output"] for r in jsonl.read(path)}
        missing = [u for u in self.order if u not in rows]
        if missing:
            raise ValueError(f"{path.name} lacks {len(missing)} held-out paragraphs, e.g. {missing[0]}")
        return [rows[u] for u in self.order]

    def reference(self) -> dict:
        return {
            "ai_input": self.score_outputs([r["ai_text"] for r in self.inputs]),
            "original": self.score_outputs([r["original"] for r in self.inputs]),
        }

    def flags(self, m: dict, original: dict) -> list[str]:
        out = []
        if m["p_human"] > original["p_human"] + OVERSHOOT_P_MARGIN or m["marker_rate"] < OVERSHOOT_MARKER_RATIO * original["marker_rate"]:
            out.append("overshoot")
        if m["number_err"] > NUMBER_ERR_LIMIT:
            out.append("number_drift")
        return out

    def article_scores(self) -> dict[str, dict]:
        """P(author) and AI-marker rate over rewritten article paragraphs, per system, plus the inputs."""
        by_system: dict[str, dict[str, list[str]]] = {}
        for f in sorted(self.cfg.work("eval", "articles", "x").parent.glob("*.blocks.jsonl")):
            m = re.match(r"(.+?)\.(.+)\.blocks\.jsonl$", f.name)
            if not m:
                continue
            log = jsonl.read(f)
            by_system.setdefault(m.group(2), {"in": [], "out": []})
            by_system[m.group(2)]["in"] += [x["input"] for x in log]
            by_system[m.group(2)]["out"] += [x["output"] for x in log]
        res = {}
        for name, d in by_system.items():
            res.setdefault("ai_input", {"p_human": float(self.clf.p_human(d["in"]).mean()),
                                        "marker_rate": marker_rate(d["in"], self.lexicon), "paragraphs": len(d["in"])})
            res[name] = {"p_human": float(self.clf.p_human(d["out"]).mean()), "marker_rate": marker_rate(d["out"], self.lexicon), "paragraphs": len(d["out"])}
        return res


def val_loss_curve(run_dir: Path) -> list[tuple[int, float, float]]:
    """(step, epoch, eval_loss) from the training log."""
    log = run_dir / "log.jsonl"
    if not log.exists():
        return []
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    return [(r["step"], r.get("epoch", 0.0), r["eval_loss"]) for r in rows if "eval_loss" in r]


def recommend_checkpoint(curve: list[tuple[int, float, float]], scored: dict[int, dict], original: dict, scorer: Scorer) -> dict:
    """Pick the checkpoint with the lowest validation loss among those without flags; explain the choice."""
    notes = []
    if curve:
        best_step, best_epoch, best_loss = min(curve, key=lambda x: x[2])
        if best_step == curve[0][0] and len(curve) > 1:
            notes.append(f"validation loss is lowest at the first evaluation (epoch {best_epoch:.2f}); train shorter or evaluate earlier")
        if curve[-1][2] > best_loss * 1.05:
            notes.append(f"validation loss rose {100 * (curve[-1][2] / best_loss - 1):.0f}% after its minimum at epoch {best_epoch:.2f}: memorisation")
    loss_at = {s: loss for s, _, loss in curve}
    candidates = []
    for step, m in sorted(scored.items()):
        fl = scorer.flags(m, original)
        candidates.append({"step": step, "val_loss": loss_at.get(step), "flags": fl, **m})
    clean = [c for c in candidates if not c["flags"]]
    pool = clean or candidates
    pick = min(pool, key=lambda c: (c["val_loss"] if c["val_loss"] is not None else float("inf"), -c["chrf"])) if pool else None
    if pool and not clean:
        notes.append("every scored checkpoint is flagged; retrain shorter or with a lower learning rate before trusting any of them")
    return {"recommended_step": pick["step"] if pick else None, "candidates": candidates, "notes": notes}
