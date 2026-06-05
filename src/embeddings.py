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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Candidate text builder
# ---------------------------------------------------------------------------

def build_candidate_text(c: dict[str, Any]) -> str:
    """
    Convert a candidate record into a single text blob for embedding.
    Weights important fields by repeating them (a simple but effective trick).
    """
    parts: list[str] = []

    # Current title and role (repeated for emphasis)
    title = c.get("current_title") or c.get("title") or ""
    if title:
        parts.append(title)
        parts.append(title)  # repeat for weight

    # Years of experience
    yoe = c.get("years_of_experience") or c.get("years_experience") or c.get("total_experience") or ""
    if yoe:
        parts.append(f"{yoe} years experience")

    # Skills (repeated for weight — most discriminative field)
    skills = c.get("skills") or []
    if isinstance(skills, list):
        skill_text = " ".join(str(s) for s in skills)
    else:
        skill_text = str(skills)
    if skill_text:
        parts.append(skill_text)
        parts.append(skill_text)  # repeat

    # Work history — titles and descriptions
    work_history = c.get("work_history") or c.get("experience") or []
    for job in work_history[:5]:  # cap at 5 to keep text manageable
        if isinstance(job, dict):
            job_title  = job.get("title") or job.get("role") or ""
            company    = job.get("company") or ""
            desc       = job.get("description") or job.get("summary") or ""
            if job_title:
                parts.append(job_title)
            if company:
                parts.append(company)
            if desc:
                parts.append(str(desc)[:300])  # truncate long descriptions

    # Education
    education = c.get("education") or []
    for edu in education[:2]:
        if isinstance(edu, dict):
            degree  = edu.get("degree") or ""
            field   = edu.get("field") or edu.get("specialization") or ""
            inst    = edu.get("institution") or edu.get("university") or ""
            parts.append(f"{degree} {field} {inst}".strip())

    # Bio / summary
    bio = c.get("bio") or c.get("summary") or c.get("about") or ""
    if bio:
        parts.append(str(bio)[:400])

    # Location
    loc = c.get("location") or c.get("city") or ""
    if loc:
        parts.append(str(loc))

    return " ".join(p for p in parts if p.strip())


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
        self.dim   = self.model.get_sentence_embedding_dimension()
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
