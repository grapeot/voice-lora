"""Build and score the blind test: the only judge of "does this sound like the author" that the model cannot game."""
from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path


def build_items(inputs: list[dict], model_out: dict[str, str], baseline_out: dict[str, str], n_s1: int = 20, n_s2: int = 20,
                seed: int = 0, min_len: int = 60, max_len: int = 420) -> dict:
    """S1: n_s1 real paragraphs + n_s1 model outputs of *different* paragraphs, shuffled ("is this the author?").
    S2: n_s2 pairs of model vs baseline on the same input, sides randomised ("which sounds more like the author?")."""
    rng = random.Random(seed)
    usable = [r for r in inputs if model_out.get(r["unit_id"]) and baseline_out.get(r["unit_id"]) and min_len <= len(r["original"]) <= max_len]
    if len(usable) < 2 * n_s1 + n_s2:
        raise ValueError(f"only {len(usable)} usable paragraphs; need {2 * n_s1 + n_s2}")
    rng.shuffle(usable)
    real, fake, pairs = usable[:n_s1], usable[n_s1 : 2 * n_s1], usable[2 * n_s1 : 2 * n_s1 + n_s2]
    s1 = [{"id": f"s1-{i:02d}", "text": r["original"], "mine": True, "unit_id": r["unit_id"]} for i, r in enumerate(real)]
    s1 += [{"id": f"s1-{i + n_s1:02d}", "text": model_out[r["unit_id"]], "mine": False, "unit_id": r["unit_id"]} for i, r in enumerate(fake)]
    rng.shuffle(s1)
    s2 = []
    for i, r in enumerate(pairs):
        model_left = rng.random() < 0.5
        m, b = model_out[r["unit_id"]], baseline_out[r["unit_id"]]
        s2.append({"id": f"s2-{i:02d}", "unit_id": r["unit_id"], "input": r["ai_text"],
                   "left": m if model_left else b, "right": b if model_left else m, "model_side": "left" if model_left else "right"})
    return {"s1": s1, "s2": s2}


def render_page(template: str, items: dict, eval_id: str, author: str, baseline_name: str) -> str:
    data = {"eval_id": eval_id, "author": author, "baseline_name": baseline_name, **items}
    return template.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))


def binom_tail(k: int, n: int, p: float = 0.5) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def score(items: dict, answers: dict) -> dict:
    s1a, s2a = answers.get("s1", {}), answers.get("s2", {})
    fake = [x for x in items["s1"] if not x["mine"]]
    real = [x for x in items["s1"] if x["mine"]]
    fake_mine = sum(s1a.get(x["id"]) is True for x in fake)
    real_mine = sum(s1a.get(x["id"]) is True for x in real)
    decided = [x for x in items["s2"] if s2a.get(x["id"]) in ("left", "right")]
    wins = sum(s2a[x["id"]] == x["model_side"] for x in decided)
    return {
        "s1_answered": sum(x["id"] in s1a for x in items["s1"]),
        "s1_model_judged_author": fake_mine, "s1_model_total": len(fake),
        "s1_real_judged_author": real_mine, "s1_real_total": len(real),
        "s2_answered": sum(x["id"] in s2a for x in items["s2"]),
        "s2_model_wins": wins, "s2_decided": len(decided),
        "s2_p_value": round(binom_tail(wins, len(decided)), 4) if decided else None,
    }


def load_answers(path: Path) -> dict:
    """Accept both the exported page JSON and {"s1": {"answers": ...}, "s2": {"answers": ...}} dumps."""
    raw = json.loads(Path(path).read_text())
    return {k: (v.get("answers", v) if isinstance(v, dict) else v) for k, v in raw.items() if k in ("s1", "s2")}


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-") or "eval"
