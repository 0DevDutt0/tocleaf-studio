"""BM25 over canonical leaf sections: a lexical baseline with no training.

Each leaf is one document, made of its breadcrumb and its own text. This matches the paper's
BM25 setup, in which the leaf node is the unit of retrieval. The implementation is small and
dependency-free so the baseline can be read and checked directly.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from stair_engine.synthetic.qa import STOPWORDS

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [token for token in _TOKEN.findall(text.lower()) if token not in STOPWORDS]


class BM25Index:
    def __init__(self, documents: dict[str, str], k1: float = 1.5, b: float = 0.75):
        if not documents:
            raise ValueError("BM25 needs at least one document")
        self.k1 = k1
        self.b = b
        self._ids = list(documents)
        self._term_counts: dict[str, Counter] = {}
        self._lengths: dict[str, int] = {}
        document_frequency: Counter = Counter()
        for doc_id, text in documents.items():
            tokens = tokenize(text)
            counts = Counter(tokens)
            self._term_counts[doc_id] = counts
            self._lengths[doc_id] = len(tokens)
            document_frequency.update(counts.keys())
        self._avg_length = sum(self._lengths.values()) / len(self._lengths) or 1.0
        n_docs = len(documents)
        self._idf = {
            term: math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)) for term, df in document_frequency.items()
        }

    def score(self, query: str, doc_id: str) -> float:
        counts = self._term_counts[doc_id]
        length = self._lengths[doc_id]
        total = 0.0
        for term in set(tokenize(query)):
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            norm = frequency * (self.k1 + 1) / (
                frequency + self.k1 * (1 - self.b + self.b * length / self._avg_length)
            )
            total += self._idf.get(term, 0.0) * norm
        return total

    def rank(self, query: str, k: int = 10) -> list[str]:
        scored = [(self.score(query, doc_id), doc_id) for doc_id in self._ids]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [doc_id for score, doc_id in scored[:k] if score > 0.0]
