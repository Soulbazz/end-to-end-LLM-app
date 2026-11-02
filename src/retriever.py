"""Retrieval engine for the TechGear Solutions corpus (Phase 2).

Three retrieval modes over ``data/corpus.json``:

* ``bm25``   - pure-Python BM25 (k1=1.5, b=0.75); strong on exact codes.
* ``vector`` - cosine similarity. Uses real embeddings (``baai/bge-m3``) when
               ``USE_API_EMBEDDINGS=1``; otherwise a deterministic offline
               char/word n-gram TF-IDF fallback that captures word stems and
               synonyms but ignores exact hyphenated code tokens.
* ``hybrid`` - Reciprocal Rank Fusion (k=60) of the BM25 and vector rankings.
"""

import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "corpus.json"

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> List[str]:
    """Lowercase; return full alphanumeric/hyphenated tokens plus hyphen sub-tokens.

    "ERR-PSU-904" -> ["err-psu-904", "err", "psu", "904"].
    """
    tokens: List[str] = []
    for tok in _TOKEN_RE.findall(text.lower()):
        tokens.append(tok)
        if "-" in tok:
            tokens.extend(tok.split("-"))
    return tokens


# --------------------------------------------------------------------------
# Offline "semantic" vectorizer helpers (fallback for vector search)
# --------------------------------------------------------------------------
_STOPWORDS = frozenset(
    "a an the and or of to in on for with is are was were be been it its this that "
    "my our your i we you do does did can could should would what which who how when "
    "if at by as from than then so not no under over into after before within any "
    "all per via has have had".split()
)

# Small synonym map: collapse related words onto a shared concept token.
_SYNONYMS = {
    "refund": "money", "credit": "money", "reimburse": "money", "reimbursement": "money",
    "fee": "money", "cost": "money", "charge": "money",
    "broken": "defect", "faulty": "defect", "damaged": "defect", "damage": "defect",
    "dead": "defect", "crushed": "defect", "defective": "defect",
    "noise": "sound", "noisy": "sound", "whine": "sound", "buzz": "sound",
    "acoustic": "sound", "vibration": "sound", "hum": "sound",
    "airplane": "air", "aircraft": "air", "plane": "air", "flight": "air",
    "battery": "lithium", "batteries": "lithium",
    "warehouse": "depot", "stockroom": "depot",
    "outage": "downtime", "outages": "downtime", "uptime": "downtime",
    "desktop": "pc", "computer": "pc", "workstation": "pc",
    "cutoff": "deadline", "limit": "deadline",
    "abroad": "international", "overseas": "international", "export": "international",
    "customs": "international",
    "restart": "boot", "bricked": "boot", "firmware": "bios",
    "shutdown": "poweroff", "tripped": "poweroff",
    "drive": "storage", "disk": "storage", "array": "storage",
    "monitor": "display", "screen": "display", "panel": "display",
    "pixel": "pixel", "pixels": "pixel",
}


def _stem(word: str) -> str:
    for suf in ("ations", "ation", "ings", "ing", "edly", "ed", "ies", "es", "s", "ly"):
        if word.endswith(suf) and len(word) - len(suf) >= 4:
            word = word[: -len(suf)]
            break
    return _SYNONYMS.get(word, word)


def _semantic_features(text: str) -> List[str]:
    """Letters-only words -> stemmed/synonym-normalised words + char trigrams.

    Any token containing a digit or hyphen (product/error codes such as
    ``err-psu-904``) is dropped entirely, mimicking how dense embeddings fail to
    pin down arbitrary identifiers.
    """
    feats: List[str] = []
    for w in _TOKEN_RE.findall(text.lower()):
        if not w.isalpha() or w in _STOPWORDS or len(w) < 3:
            continue
        s = _stem(w)
        feats.append("w:" + s)
        padded = f"#{s}#"
        feats.extend("c:" + padded[i:i + 3] for i in range(len(padded) - 2))
    return feats


def _cosine(a: Dict[Any, float], b: Dict[Any, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def rrf(rank_lists: List[List[str]], k: int = 60) -> List[str]:
    """Reciprocal Rank Fusion: score(d) = sum 1 / (k + rank), 1-based ranks.

    Ties are broken by first appearance across the input lists.
    """
    scores: Dict[str, float] = {}
    first_seen: Dict[str, int] = {}
    for ranks in rank_lists:
        for pos, doc_id in enumerate(ranks, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + pos)
            first_seen.setdefault(doc_id, len(first_seen))
    return sorted(scores, key=lambda d: (-scores[d], first_seen[d]))


class Retriever:
    """Loads the corpus and exposes BM25, vector and hybrid search."""

    def __init__(self, corpus_path: Optional[Path] = None, k1: float = 1.5, b: float = 0.75) -> None:
        with open(corpus_path or CORPUS_PATH, encoding="utf-8") as f:
            self.docs: List[Dict[str, Any]] = json.load(f)
        self.by_id = {d["id"]: d for d in self.docs}
        self.k1, self.b = k1, b

        # BM25 index over title + text + category
        self._doc_texts = [f'{d["title"]} {d["text"]} {d["category"]}' for d in self.docs]
        self._doc_tf = [Counter(tokenize(t)) for t in self._doc_texts]
        self._doc_len = [sum(tf.values()) for tf in self._doc_tf]
        self._avgdl = sum(self._doc_len) / len(self.docs)
        self._df: Counter = Counter()
        for tf in self._doc_tf:
            self._df.update(tf.keys())

        # Vector index (lazily built)
        self._tfidf_vecs: Optional[List[Dict[str, float]]] = None
        self._idf: Dict[str, float] = {}
        self._emb: Optional[List[List[float]]] = None
        self._api_failed = False

    # ---------------------------- BM25 ----------------------------------
    def _bm25_score(self, q_tokens: List[str], i: int) -> float:
        n = len(self.docs)
        tf, dl = self._doc_tf[i], self._doc_len[i]
        score = 0.0
        for t in q_tokens:
            f = tf.get(t, 0)
            if not f:
                continue
            df = self._df[t]
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
            score += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self._avgdl))
        return score

    def search_bm25(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q_tokens = tokenize(query)
        scored = [(self._bm25_score(q_tokens, i), i) for i in range(len(self.docs))]
        scored = [(s, i) for s, i in scored if s > 0]
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [self._with_score(self.docs[i], s) for s, i in scored[:top_k]]

    # ---------------------------- Vector --------------------------------
    def _build_tfidf(self) -> None:
        feats = [_semantic_features(t) for t in self._doc_texts]
        n = len(feats)
        df: Counter = Counter()
        for f in feats:
            df.update(set(f))
        self._idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
        self._tfidf_vecs = [self._weight(f) for f in feats]

    def _weight(self, feats: List[str]) -> Dict[str, float]:
        tf = Counter(feats)
        return {t: (1 + math.log(c)) * self._idf[t] for t, c in tf.items() if t in self._idf}

    def _api_embed(self, texts: List[str]) -> Optional[List[List[float]]]:
        if self._api_failed:
            return None
        try:
            from openai import OpenAI

            resp = OpenAI().embeddings.create(model="baai/bge-m3", input=texts)
            return [item.embedding for item in resp.data]
        except Exception as exc:  # offline / no endpoint -> fallback
            self._api_failed = True
            print(f"[retriever] API embeddings unavailable ({type(exc).__name__}); using offline TF-IDF fallback.")
            return None

    def _vector_scores(self, query: str) -> List[float]:
        if os.getenv("USE_API_EMBEDDINGS") == "1":
            if self._emb is None:
                self._emb = self._api_embed(self._doc_texts)
            q = self._api_embed([query]) if self._emb is not None else None
            if self._emb is not None and q is not None:
                qv = dict(enumerate(q[0]))
                return [_cosine(qv, dict(enumerate(e))) for e in self._emb]
        if self._tfidf_vecs is None:
            self._build_tfidf()
        qv = self._weight(_semantic_features(query))
        return [_cosine(qv, dv) for dv in self._tfidf_vecs]

    def search_vector(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        scores = self._vector_scores(query)
        order = [i for i in sorted(range(len(self.docs)), key=lambda i: (-scores[i], i)) if scores[i] > 0]
        return [self._with_score(self.docs[i], scores[i]) for i in order[:top_k]]

    # ---------------------------- Hybrid --------------------------------
    def search_hybrid(self, query: str, top_k: int = 5, rrf_k: int = 60) -> List[Dict[str, Any]]:
        bm25_ids = [d["id"] for d in self.search_bm25(query, top_k=20)]
        vec_ids = [d["id"] for d in self.search_vector(query, top_k=20)]
        fused = rrf([bm25_ids, vec_ids], k=rrf_k)[:top_k]
        return [dict(self.by_id[i], citation=f"[{i}]") for i in fused]

    def search(self, query: str, mode: str = "hybrid", top_k: int = 5) -> List[Dict[str, Any]]:
        """Dispatch to ``bm25`` / ``vector`` / ``hybrid``."""
        if mode == "bm25":
            return self.search_bm25(query, top_k)
        if mode == "vector":
            return self.search_vector(query, top_k)
        if mode == "hybrid":
            return self.search_hybrid(query, top_k)
        raise ValueError(f"Unknown retrieval mode: {mode}")

    @staticmethod
    def _with_score(doc: Dict[str, Any], score: float) -> Dict[str, Any]:
        return dict(doc, score=round(score, 6), citation=f'[{doc["id"]}]')


_default: Optional[Retriever] = None


def _get() -> Retriever:
    global _default
    if _default is None:
        _default = Retriever()
    return _default


def search_bm25(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    return _get().search_bm25(query, top_k)


def search_vector(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    return _get().search_vector(query, top_k)


def search_hybrid(query: str, top_k: int = 5, rrf_k: int = 60) -> List[Dict[str, Any]]:
    return _get().search_hybrid(query, top_k, rrf_k)


def format_passages_for_prompt(docs: List[Dict[str, Any]]) -> str:
    """Format docs as ``[doc_01] (Title): text`` lines for the agent prompt."""
    return "\n\n".join(f'[{d["id"]}] ({d["title"]}): {d["text"]}' for d in docs)
