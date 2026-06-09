"""
embeddings.py — DeepShortlist
Sentence-transformer embedding engine with FAISS index for fast
first-pass candidate retrieval on CPU within compute budget.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.jd_config import (
    EMBEDDING_BATCH,
    EMBEDDING_MODEL,
    FAISS_FIRST_PASS,
    JD_TEXT,
)
from src.field_extractor import build_rich_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Candidate text builder
# ---------------------------------------------------------------------------

def build_candidate_text(c: dict[str, Any]) -> str:
    """
    Convert a candidate record into a single text blob for embedding.
    Uses the universal field_extractor.
    """
    return build_rich_text(c)


# ---------------------------------------------------------------------------
# Embedding Engine
# ---------------------------------------------------------------------------

class EmbeddingEngine:
    """
    Wraps SentenceTransformer + FAISS for fast approximate retrieval.

    Usage
    -----
    engine = EmbeddingEngine()
    engine.build_index(candidates)
    top_indices, top_scores = engine.query(top_k=2000)
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        logger.info(f"Loading embedding model: {model_name}")
        t0 = time.time()
        self.model = SentenceTransformer(model_name)
        # Handle different sentence-transformers versions
        if hasattr(self.model, 'get_sentence_embedding_dimension'):
            self.dim = self.model.get_sentence_embedding_dimension()
        else:
            self.dim = self.model.get_embedding_dimension()
        logger.info(f"Model loaded in {time.time()-t0:.1f}s  (dim={self.dim})")

        self._jd_vec: np.ndarray | None = None
        self._index:  faiss.IndexFlatIP | None = None
        self._n_candidates: int = 0

    # ------------------------------------------------------------------
    def encode_jd(self, jd_text: str | None = None) -> np.ndarray:
        text = jd_text or JD_TEXT
        vec  = self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        self._jd_vec = vec.reshape(1, -1).astype("float32")
        logger.info("JD embedding computed")
        return self._jd_vec

    # ------------------------------------------------------------------
    def build_index(
        self,
        candidates: list[dict[str, Any]],
        jd_text: str | None = None,
    ) -> None:
        """
        Build FAISS IndexFlatIP (inner-product = cosine on normalised vecs).
        All candidate vectors are stored in RAM (~384 dims × 100K × 4 bytes ≈ 154 MB).
        """
        logger.info(f"Encoding {len(candidates):,} candidates in batches of {EMBEDDING_BATCH}…")
        t0 = time.time()

        texts   = [build_candidate_text(c) for c in candidates]
        all_vecs: list[np.ndarray] = []

        for start in range(0, len(texts), EMBEDDING_BATCH):
            batch = texts[start : start + EMBEDDING_BATCH]
            vecs  = self.model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=64,
            )
            all_vecs.append(vecs.astype("float32"))

            done = min(start + EMBEDDING_BATCH, len(texts))
            if done % 10_000 == 0 or done == len(texts):
                elapsed = time.time() - t0
                rate    = done / elapsed if elapsed > 0 else 0
                logger.info(f"  Encoded {done:,}/{len(texts):,}  ({rate:.0f} cands/s)")

        matrix = np.vstack(all_vecs)  # (N, dim)
        logger.info(f"All encodings done in {time.time()-t0:.1f}s")

        # Build FAISS index
        self._index        = faiss.IndexFlatIP(self.dim)
        self._index.add(matrix)
        self._n_candidates = len(candidates)
        logger.info(f"FAISS index built: {self._index.ntotal:,} vectors")

        # Encode JD
        self.encode_jd(jd_text)

    # ------------------------------------------------------------------
    def query(self, top_k: int = FAISS_FIRST_PASS) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (indices, cosine_scores) for top_k candidates.
        Indices correspond to the original candidates list order.
        """
        if self._index is None or self._jd_vec is None:
            raise RuntimeError("Call build_index() before query()")

        k = min(top_k, self._n_candidates)
        scores, indices = self._index.search(self._jd_vec, k)
        return indices[0], scores[0]   # flatten batch dimension

    # ------------------------------------------------------------------
    def candidate_similarity(self, candidate: dict[str, Any]) -> float:
        """Compute cosine similarity for a single candidate (used in re-scoring)."""
        text = build_candidate_text(candidate)
        vec  = self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        vec  = vec.reshape(1, -1).astype("float32")
        sim  = float(np.dot(vec, self._jd_vec.T).squeeze())
        return max(0.0, min(1.0, sim))
