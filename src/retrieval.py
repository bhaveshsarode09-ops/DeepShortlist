"""
retrieval.py — DeepShortlist v2
Two-stage offline retrieval:
  Stage A: Field-weighted BM25+ (100K → 3000)  ~5s
  Stage B: TF-IDF + LSA cosine similarity      ~3s

BM25+ (Lv & Zhai, 2011): adds lower-bound δ to prevent zero scores.
LSA (Deerwester et al.): captures latent semantic structure via SVD.
No external models. No internet. Pure sklearn + numpy.
"""
from __future__ import annotations
import logging, math, re, time
from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from src.schema import Schema, JD_TEXT

logger = logging.getLogger(__name__)

# ── Tokeniser ─────────────────────────────────────────────────────────────────

_STOP = frozenset({
    "the","a","an","and","or","of","in","on","at","to","for","with","is",
    "was","are","be","been","have","has","that","this","it","we","our","you",
    "as","by","from","not","but","if","so","do","did","can","will","would",
    "should","may","might","also","more","than","over","into","been","after",
    "before","about","between","through","during","above","below","up","down",
})

def tokenise(text: str) -> list[str]:
    if not text: return []
    tokens = re.findall(r"[a-z0-9][a-z0-9\-/]*[a-z0-9]|[a-z0-9]", text.lower())
    return [t for t in tokens if t not in _STOP and len(t) > 1]


# ── Field-weighted BM25+ ──────────────────────────────────────────────────────

class BM25Plus:
    """
    Okapi BM25+ with field weighting.
    k1=1.6, b=0.75, delta=0.5

    Field weights applied before IDF computation:
      title   × 3.0
      skills  × 2.5
      rest    × 1.0
    """
    K1 = 1.6
    B  = 0.75
    D  = 0.5      # BM25+ delta (lower bound)

    def __init__(self):
        self._idf:      dict[str, float] = {}
        self._tf_docs:  list[dict[str, float]] = []
        self._dl:       np.ndarray | None = None
        self._avgdl:    float = 0.0
        self._N:        int   = 0

    def fit(self, corpus: list[list[str]]) -> "BM25Plus":
        self._N = len(corpus)
        dl = np.array([len(d) for d in corpus], dtype=np.float32)
        self._dl     = dl
        self._avgdl  = float(dl.mean()) if len(dl) else 1.0

        df: dict[str, int] = defaultdict(int)
        self._tf_docs = []
        for doc in corpus:
            freq: dict[str, float] = defaultdict(float)
            for t in doc:
                freq[t] += 1.0
            self._tf_docs.append(dict(freq))
            for t in set(doc):
                df[t] += 1

        # BM25+ IDF: log((N+1)/(df+0.5))
        self._idf = {
            t: math.log((self._N + 1) / (cnt + 0.5))
            for t, cnt in df.items()
        }
        return self

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        """Vectorised BM25+ scores for all documents."""
        out = np.zeros(self._N, dtype=np.float32)
        k1, b, avg, D = self.K1, self.B, self._avgdl, self.D
        dl = self._dl

        for q in set(query_tokens):
            idf = self._idf.get(q, 0.0)
            if idf <= 0:
                continue
            tf_arr = np.array(
                [self._tf_docs[i].get(q, 0.0) for i in range(self._N)],
                dtype=np.float32,
            )
            norm   = k1 * (1 - b + b * dl / avg)
            out   += idf * ((tf_arr * (k1 + 1)) / (tf_arr + norm) + D)

        return out


# ── Field-weighted corpus builder ─────────────────────────────────────────────

def _weighted_tokens(schema: Schema, c: dict) -> list[str]:
    """
    Build a weighted token list:
      title  repeated 3×
      skills repeated 2.5× (rounded to 2×)
      rest   1×
    """
    parts: list[str] = []
    title  = schema.title(c)
    skills = schema.skills(c)

    if title:
        tok = tokenise(title)
        parts += tok * 3

    if skills:
        tok = tokenise(" ".join(skills))
        parts += tok * 2

    for job in schema.work_history(c)[:3]:
        role = (job.get("title") or job.get("role") or
                job.get("designation") or "")
        desc = (job.get("description") or job.get("responsibilities") or
                job.get("summary") or "")
        if role: parts += tokenise(role)
        if desc: parts += tokenise(str(desc)[:200])

    bio = (c.get("bio") or c.get("summary") or c.get("about") or "")
    if bio: parts += tokenise(str(bio)[:200])

    return parts


# ── LSA engine ────────────────────────────────────────────────────────────────

class LSAEngine:
    """
    TF-IDF + TruncatedSVD (LSA).
    Captures latent semantic relationships missed by BM25.
    n_components=120 balances quality vs speed.
    """
    N_COMPONENTS = 120

    def __init__(self):
        self._tfidf = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=40_000,
            sublinear_tf=True,
            min_df=2,
        )
        self._svd    = TruncatedSVD(n_components=self.N_COMPONENTS, random_state=42)
        self._matrix: np.ndarray | None = None
        self._jd_vec: np.ndarray | None = None

    def fit(self, texts: list[str], jd_text: str) -> None:
        all_texts = [jd_text] + texts
        tfidf_mat = self._tfidf.fit_transform(all_texts)
        n_feats = tfidf_mat.shape[1]
        k = min(self.N_COMPONENTS, n_feats - 1, len(texts))
        if k != self._svd.n_components:
            self._svd = TruncatedSVD(n_components=max(k,2), random_state=42)
        lsa_mat   = self._svd.fit_transform(tfidf_mat)
        lsa_mat   = normalize(lsa_mat, norm="l2")
        self._jd_vec = lsa_mat[0:1]
        self._matrix = lsa_mat[1:]

    def scores(self) -> np.ndarray:
        return cosine_similarity(self._jd_vec, self._matrix)[0]

    def score_single(self, text: str) -> float:
        v = self._tfidf.transform([text])
        v = self._svd.transform(v)
        v = normalize(v, norm="l2")
        return float(cosine_similarity(self._jd_vec, v)[0, 0])


# ── Main retrieval engine ─────────────────────────────────────────────────────

class RetrievalEngine:
    """
    Two-stage retrieval:
      1. BM25+ first pass  (100K → top_first_pass, default 3000)
      2. LSA reranking     (3000 → return all, scored)

    Returns ranked (indices, scores) over the full candidate set.
    """
    FIRST_PASS = 3000

    def __init__(self, schema: Schema, first_pass: int = FIRST_PASS):
        self.schema     = schema
        self.first_pass = first_pass
        self._bm25      = BM25Plus()
        self._lsa       = LSAEngine()
        self._jd_tokens = tokenise(JD_TEXT)
        self._n:         int = 0
        self._bm25_norm: np.ndarray | None = None
        self._lsa_scores_full: np.ndarray | None = None

    def build(self, candidates: list[dict]) -> None:
        self._n = len(candidates)
        t0 = time.time()
        logger.info(f"Building BM25+ corpus for {self._n:,} candidates…")

        # BM25+ corpus (field-weighted tokens)
        corpus = [_weighted_tokens(self.schema, c) for c in candidates]
        self._bm25.fit(corpus)
        bm25_raw = self._bm25.scores(self._jd_tokens)
        bm25_max = bm25_raw.max()
        self._bm25_norm = bm25_raw / bm25_max if bm25_max > 0 else bm25_raw
        logger.info(f"BM25+ done in {time.time()-t0:.1f}s  max={bm25_max:.2f}")

        # LSA on top-FIRST_PASS candidates only
        t1 = time.time()
        fp = min(self.first_pass, self._n)
        top_idx = np.argsort(self._bm25_norm)[::-1][:fp]
        fp_texts = [self.schema.build_text(candidates[i]) for i in top_idx]
        self._lsa.fit(fp_texts, JD_TEXT)
        lsa_fp_scores = self._lsa.scores()

        # Map LSA scores back to full index space
        self._lsa_scores_full = np.zeros(self._n, dtype=np.float32)
        for rank_in_fp, orig_idx in enumerate(top_idx):
            self._lsa_scores_full[orig_idx] = lsa_fp_scores[rank_in_fp]

        logger.info(f"LSA done in {time.time()-t1:.1f}s")

    def combined_scores(self, w_bm25: float = 0.55, w_lsa: float = 0.45) -> np.ndarray:
        """Weighted combination of BM25+ and LSA scores."""
        return w_bm25 * self._bm25_norm + w_lsa * self._lsa_scores_full

    def top_indices(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        combined = self.combined_scores()
        idx = np.argsort(combined)[::-1][:k]
        return idx, combined[idx]
