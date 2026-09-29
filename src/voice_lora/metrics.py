"""Automatic metrics: how far an output moved from AI voice toward the author's, and whether it kept the facts.

Caveat: the voice classifier and the AI-marker lexicon are learned from the same (AI rewrite, original)
pairs the LoRA is trained on, so the LoRA is naturally good at fooling them. Use them to compare
checkpoints and baselines; judge "does it sound like the author" with a blind test.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
from sacrebleu.metrics import CHRF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .pairs import keep_rewrite, number_ok

CJK_RE = re.compile(r"[一-鿿]")
_chrf = CHRF(word_order=2)


def train_material(units: list[dict], rewrites: list[dict]) -> tuple[list[str], list[str]]:
    """(author paragraphs, kept AI rewrites) from the train split only."""
    by_id = {u["unit_id"]: u for u in units}
    human = [u["text"] for u in units if u["split"] == "train"]
    ai = []
    for r in rewrites:
        u = by_id.get(r["unit_id"])
        if u and u["split"] == "train" and r.get("rewrite") and keep_rewrite(u["text"], r["rewrite"])[0]:
            ai.append(r["rewrite"])
    return human, ai


# Typographic variants that say nothing about voice: which quote glyph or full/half-width form a text uses
# is an editor or keyboard setting. Left in, they were among the classifier's strongest features (the
# author's posts use straight quotes, AI output curly ones), so swapping quotes alone raised the score.
_PUNCT = str.maketrans({"“": '"', "”": '"', "„": '"', "＂": '"', "‘": "'", "’": "'", "＇": "'",
                        "；": ";", "：": ":", "，": ",", "（": "(", "）": ")", "！": "!", "？": "?"})


def normalize_punct(text: str) -> str:
    return text.translate(_PUNCT)


class VoiceClassifier:
    """Char 1-3gram logistic regression: P(written by the author) vs P(AI rewrite), on punctuation-normalized text."""

    def __init__(self, human: list[str], ai: list[str]) -> None:
        self.vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True, min_df=2, preprocessor=normalize_punct)
        x = self.vec.fit_transform(human + ai)
        y = np.array([1] * len(human) + [0] * len(ai))
        self.clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced").fit(x, y)

    def p_human(self, texts: list[str]) -> np.ndarray:
        return self.clf.predict_proba(self.vec.transform(texts))[:, 1]

    @classmethod
    def from_state(cls, vocabulary: dict[str, int], idf: list[float], coef: list[float], intercept: float) -> VoiceClassifier:
        """Rebuild a fitted classifier from its saved parameters (see classify.Detector.save)."""
        self = cls.__new__(cls)
        self.vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True, preprocessor=normalize_punct, vocabulary=vocabulary)
        self.vec.idf_ = np.asarray(idf, dtype=float)
        lr = LogisticRegression()
        lr.coef_ = np.asarray([coef], dtype=float)
        lr.intercept_ = np.asarray([intercept], dtype=float)
        lr.classes_ = np.array([0, 1])
        lr.n_features_in_ = len(coef)
        self.clf = lr
        return self


def ai_marker_lexicon(human: list[str], ai: list[str], top: int = 300, min_count: int = 20) -> list[str]:
    """Chinese character n-grams (2-4) that AI rewrites use far more than the author, by smoothed log-odds."""

    def grams(texts: list[str]) -> Counter:
        c: Counter = Counter()
        for t in texts:
            t = re.sub(r"\s", "", t)
            for n in (2, 3, 4):
                c.update(t[i : i + n] for i in range(len(t) - n + 1) if all(CJK_RE.match(ch) for ch in t[i : i + n]))
        return c

    ch, ca = grams(human), grams(ai)
    nh, na = max(1, sum(ch.values())), max(1, sum(ca.values()))
    scored = []
    for g, a in ca.items():
        if a < min_count:
            continue
        h = ch.get(g, 0)
        scored.append((math.log((a + 1) / na) - math.log((h + 1) / nh), g))
    scored.sort(reverse=True)
    lexicon: list[str] = []
    for _, g in scored:
        if not any(g in x or x in g for x in lexicon):
            lexicon.append(g)
        if len(lexicon) >= top:
            break
    return lexicon


def marker_rate(texts: list[str], lexicon: list[str]) -> float:
    """AI-marker hits per 1,000 characters."""
    hits = sum(t.count(g) for t in texts for g in lexicon)
    return 1000 * hits / max(1, sum(len(t) for t in texts))


def chrf(outputs: list[str], refs: list[str]) -> float:
    return _chrf.corpus_score(outputs, [refs]).score


GROUPED_NUM_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


def _canon_numbers(text: str) -> set[str]:
    """Numbers by value; "14,000" and "14000" are the same number."""
    return {f"{float(n.replace(',', '').rstrip('.')):g}" for n in GROUPED_NUM_RE.findall(text)}


def _same_year(a: str, b_set: set[str]) -> bool:
    """"22年" and "2022年" name the same year."""
    v = float(a)
    return (0 <= v < 100 and f"{2000 + v:g}" in b_set) or (2000 <= v < 2100 and f"{v - 2000:g}" in b_set)


def numbers_changed(inp: str, out: str) -> set[str]:
    """Numbers dropped, altered or invented by `out` relative to `inp` (formatting differences ignored)."""
    a, b = _canon_numbers(inp), _canon_numbers(out)
    diff = {n for n in a - b if not number_ok(n, out) and not _same_year(n, b)}
    return diff | {n for n in b - a if not number_ok(n, inp) and not _same_year(n, a)}


def number_errors(inputs: list[str], outputs: list[str]) -> float:
    """Share of outputs that drop, alter or invent a number relative to their input.

    Numbers compare by value (100.0 == 100, 22 == 2022 as a year); 1-10 may switch to Chinese numerals.
    Latin tokens are not counted: restoring an English term that the AI input had translated is part
    of the target voice, and the author's own originals would score about 7% otherwise.
    """
    bad = sum(bool(numbers_changed(i, o)) for i, o in zip(inputs, outputs, strict=True))
    return bad / max(1, len(inputs))


def score_system(outputs: list[str], inputs: list[str], refs: list[str], clf: VoiceClassifier, lexicon: list[str]) -> dict:
    p = clf.p_human(outputs)
    return {
        "chrf": chrf(outputs, refs),
        "p_human": float(p.mean()),
        "share_human": float((p > 0.5).mean()),
        "marker_rate": marker_rate(outputs, lexicon),
        "number_err": number_errors(inputs, outputs),
        "len_ratio": float(np.median([len(o) / max(1, len(r)) for o, r in zip(outputs, refs, strict=True)])),
    }
