"""The voice classifier as a standalone tool: fit it from a run's data, check how far it can be trusted,
score markdown articles, and save/load it as plain JSON (no pickle, so a shared file cannot run code).

The classifier only sees wording (character n-grams). It cannot see invented facts, dropped arguments or
structure; see docs/classifier.md before reading much into a single score.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from . import jsonl
from .config import Config
from .mdproc import split_blocks
from .metrics import VoiceClassifier, ai_marker_lexicon, marker_rate, train_material
from .pairs import keep_rewrite

LENGTH_BUCKETS = [(0, 80), (80, 200), (200, 10**9)]
MIN_PARAGRAPH = 20   # shorter prose blocks ("原稿：", captions) carry almost no signal
FORMAT = "voice-lora-classifier/1"
BUNDLED = "classifier-v1.1.json"   # shipped in voice_lora/data; see docs/classifier.md
# Article-level bands, from the calibration in docs/classifier.md: the author's own posts sit around 0.9, other
# people's human writing mostly 0.35-0.8, AI-drafted text mostly below 0.3. Upper bounds, checked in order.
BANDS = [(0.3, "AI 味浓"), (0.5, "灰区"), (0.85, "像真人写的"), (float("inf"), "接近作者本人")]


def band(p: float, bands: list[tuple[float, str]] = BANDS) -> str:
    return next(label for upper, label in bands if p < upper)


def parse_bands(spec: str) -> list[tuple[float, str]]:
    """"0.3,0.5,0.85" -> BANDS with those three boundaries."""
    cuts = [float(x) for x in spec.split(",")]
    if len(cuts) != len(BANDS) - 1 or cuts != sorted(cuts):
        raise ValueError(f"--bands needs {len(BANDS) - 1} increasing numbers, e.g. 0.3,0.5,0.85")
    return [(c, label) for c, (_, label) in zip([*cuts, float("inf")], BANDS, strict=True)]


@dataclass
class Detector:
    """A fitted classifier plus the AI-marker lexicon learned from the same material."""
    clf: VoiceClassifier
    lexicon: list[str]
    author: str = "作者"

    def save(self, path: str | Path) -> None:
        vec, lr = self.clf.vec, self.clf.clf
        data = {"format": FORMAT, "author": self.author, "lexicon": self.lexicon,
                "vocabulary": {k: int(v) for k, v in vec.vocabulary_.items()}, "idf": vec.idf_.tolist(),
                "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def bundled(cls) -> Detector:
        """The released classifier that ships with the package (top 10,000 features, v1.1 data)."""
        with resources.as_file(resources.files("voice_lora") / "data" / BUNDLED) as p:
            return cls.load(p)

    def prune(self, top: int) -> Detector:
        """Keep the `top` features with the largest weights. On v1.1 data 10,000 of 667k features keep the
        held-out AUC (0.965 vs 0.967) and shrink the saved file from 39 MB to 0.55 MB."""
        vec, lr = self.clf.vec, self.clf.clf
        coef = lr.coef_[0]
        keep = np.argsort(-np.abs(coef))[:top]
        inv = {i: t for t, i in vec.vocabulary_.items()}
        clf = VoiceClassifier.from_state({inv[i]: n for n, i in enumerate(keep)}, vec.idf_[keep].tolist(),
                                         coef[keep].tolist(), float(lr.intercept_[0]))
        return Detector(clf, self.lexicon, self.author)

    @classmethod
    def load(cls, path: str | Path) -> Detector:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != FORMAT:
            raise ValueError(f"{path} is not a {FORMAT} file")
        return cls(VoiceClassifier.from_state(data["vocabulary"], data["idf"], data["coef"], data["intercept"]),
                   data["lexicon"], data.get("author", "作者"))


def fit(cfg: Config) -> tuple[Detector, list[dict], list[dict]]:
    """Fit on the train split of a run (units.jsonl + rewrites.jsonl in its workdir); also returns units and rewrites."""
    units = jsonl.read(cfg.work("units.jsonl"))
    rewrites = jsonl.read(cfg.work("rewrites.jsonl"))
    human, ai = train_material(units, rewrites)
    return Detector(VoiceClassifier(human, ai), ai_marker_lexicon(human, ai), cfg.get("author.name", "作者")), units, rewrites


def separation(clf: VoiceClassifier, human: list[str], ai: list[str]) -> dict:
    ph, pa = clf.p_human(human), clf.p_human(ai)
    y = np.r_[np.ones(len(ph)), np.zeros(len(pa))]
    return {"n_author": len(ph), "n_ai": len(pa), "auc": float(roc_auc_score(y, np.r_[ph, pa])),
            "author_mean": float(ph.mean()), "ai_mean": float(pa.mean()),
            "author_above_half": float((ph > 0.5).mean()), "ai_below_half": float((pa < 0.5).mean())}


def check(clf: VoiceClassifier, units: list[dict], rewrites: list[dict]) -> dict[str, dict]:
    """How well the classifier separates the author's held-out paragraphs (val + test splits) from their AI
    rewrites, overall, per rewriter model ("first" = plain prompt ids) and per paragraph length."""
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
    return rows


def check_table(rows: dict[str, dict]) -> str:
    lines = ["| 切片 | 原文段 | AI 段 | AUC | 原文均值 | AI 均值 | 原文 >0.5 | AI <0.5 |", "|---|---|---|---|---|---|---|---|"]
    for k, m in rows.items():
        lines.append(f"| {k} | {m['n_author']} | {m['n_ai']} | {m['auc']:.3f} | {m['author_mean']:.3f} | {m['ai_mean']:.3f} "
                     f"| {m['author_above_half']:.0%} | {m['ai_below_half']:.0%} |")
    return "\n".join(lines)


def paragraphs(md: str, min_chars: int = MIN_PARAGRAPH) -> list[str]:
    """Prose paragraphs of a markdown article: no front matter, headings, quotes, tables, code or short blocks."""
    md = re.sub(r"\A---\n.*?\n---\n", "", md, flags=re.S)
    return [b.plain.strip() for b in split_blocks(md) if b.rewrite and len(b.plain.strip()) >= min_chars]


def markers_in(text: str, lexicon: list[str]) -> list[str]:
    return [g for g in lexicon if g in text]


def score_files(det: Detector, files: list[str | Path], min_chars: int = MIN_PARAGRAPH, worst: int = 0,
                bands: list[tuple[float, str]] = BANDS) -> list[dict]:
    """P(author) per article (length-weighted mean over its prose paragraphs), its band, and AI markers per
    1,000 characters, sorted from most AI-like to most author-like. With `worst`, each row also lists that many
    lowest-scoring paragraphs and the AI markers they contain. Files without prose are skipped.

    Read the article-level number; single paragraphs are noisy (half of other people's human paragraphs score
    below 0.5)."""
    rows = []
    for f in files:
        paras = paragraphs(Path(f).read_text(encoding="utf-8"), min_chars)
        if not paras:
            continue
        p = det.clf.p_human(paras)
        w = np.array([len(t) for t in paras], dtype=float)
        pa = float((p * w).sum() / w.sum())
        row = {"file": str(f), "p_author": pa, "band": band(pa, bands), "marker_rate": marker_rate(paras, det.lexicon),
               "paragraphs": len(paras), "chars": int(w.sum())}
        if worst:
            row["worst"] = [{"p": float(p[i]), "text": paras[i], "markers": markers_in(paras[i], det.lexicon)}
                            for i in np.argsort(p)[:worst]]
        rows.append(row)
    rows.sort(key=lambda r: r["p_author"])
    return rows


def score_table(rows: list[dict], author: str = "作者") -> str:
    lines = [f"| 像{author}写的概率 | 判断 | AI 腔词/千字 | 段数 | 文件 |", "|---|---|---|---|---|"]
    lines += [f"| {r['p_author']:.3f} | {r['band']} | {r['marker_rate']:.2f} | {r['paragraphs']} | {r['file']} |" for r in rows]
    for r in rows:
        if r.get("worst"):
            lines += ["", f"{r['file']}：分数最低的段落"]
            for x in r["worst"]:
                hint = f"（AI 腔词：{'、'.join(x['markers'][:6])}）" if x["markers"] else ""
                lines.append(f"- {x['p']:.2f} {x['text'][:80]}{'…' if len(x['text']) > 80 else ''}{hint}")
    return "\n".join(lines)
