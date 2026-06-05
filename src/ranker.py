"""
ranker.py — DeepShortlist
Orchestrates the full pipeline:
  1. Encode JD and all candidates with FAISS first-pass
  2. Re-score top-N with full multi-signal scorer
  3. Detect and eliminate honeypots
  4. Final ranking + reasoning generation
  5. Return validated DataFrame ready for CSV export
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from src.embeddings import EmbeddingEngine
from src.honeypot import HoneypotDetector
from src.jd_config import FAISS_FIRST_PASS, FINAL_TOP_K, JD_TEXT, WEIGHTS
from src.reasoning import generate_reasoning
from src.scorer import compute_full_score

logger = logging.getLogger(__name__)


class DeepShortlistRanker:
    """
    End-to-end ranking system for the Redrob Intelligent Candidate
    Discovery & Ranking challenge.

    Parameters
    ----------
    model_name : str
        Sentence-transformers model (default: all-MiniLM-L6-v2).
    jd_text : str | None
        Override JD text; uses embedded JD_TEXT if None.
    faiss_first_pass : int
        How many candidates to retrieve in FAISS stage before full scoring.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        jd_text: str | None = None,
        faiss_first_pass: int = FAISS_FIRST_PASS,
    ) -> None:
        self.model_name       = model_name
        self.jd_text          = jd_text or JD_TEXT
        self.faiss_first_pass = faiss_first_pass
        self.engine           = EmbeddingEngine(model_name=model_name)
        self.honeypot_detector= HoneypotDetector(suspicion_threshold=0.60)

    # ------------------------------------------------------------------
    def rank(
        self,
        candidates: list[dict[str, Any]],
        top_k: int = FINAL_TOP_K,
    ) -> pd.DataFrame:
        """
        Main entry point. Returns a DataFrame with columns:
          candidate_id, rank, score, reasoning
        Sorted by rank ascending (1 = best).
        """
        total_start = time.time()
        n = len(candidates)
        logger.info(f"Starting DeepShortlist pipeline on {n:,} candidates")

        # ---- Stage 1: Build FAISS index and get first-pass top candidates ---
        logger.info("Stage 1/4: Building embedding index…")
        self.engine.build_index(candidates, jd_text=self.jd_text)

        first_pass_k = min(self.faiss_first_pass, n)
        logger.info(f"Stage 1/4: FAISS query → top {first_pass_k:,}…")
        fp_indices, fp_scores = self.engine.query(top_k=first_pass_k)

        logger.info(
            f"FAISS first-pass: top semantic score={fp_scores[0]:.4f}, "
            f"bottom={fp_scores[-1]:.4f}"
        )

        # ---- Stage 2: Full multi-signal scoring on first-pass pool -----
        logger.info(f"Stage 2/4: Full scoring of {len(fp_indices):,} candidates…")
        scored_records = []
        t2 = time.time()

        for i, (idx, sem_score) in enumerate(zip(fp_indices, fp_scores)):
            c = candidates[int(idx)]

            # Honeypot check
            hp_suspicion = self.honeypot_detector.score(c)

            # Full score
            scores = compute_full_score(c, float(sem_score), hp_suspicion)

            scored_records.append({
                "candidate_id": c.get("candidate_id", f"UNKNOWN_{idx}"),
                "_candidate":   c,
                "_scores":      scores,
                "_final":       scores["final_score"],
            })

            if (i + 1) % 500 == 0:
                logger.info(f"  Scored {i+1:,}/{len(fp_indices):,}")

        logger.info(f"Stage 2/4 done in {time.time()-t2:.1f}s")

        # ---- Stage 3: Sort, deduplicate, take top_k -------------------
        logger.info("Stage 3/4: Ranking and deduplication…")
        scored_records.sort(key=lambda r: r["_final"], reverse=True)

        # Deduplicate candidate_ids (FAISS can return same id if index rebuilt)
        seen_ids: set[str] = set()
        deduped = []
        for r in scored_records:
            cid = r["candidate_id"]
            if cid not in seen_ids:
                seen_ids.add(cid)
                deduped.append(r)

        top_records = deduped[:top_k]
        logger.info(
            f"Top {top_k} selected: "
            f"score range [{top_records[-1]['_final']:.4f} – {top_records[0]['_final']:.4f}]"
        )

        # Log honeypot stats
        n_honeypots = sum(1 for r in scored_records if r["_scores"]["honeypot_suspicion"] >= 0.60)
        logger.info(f"Honeypots detected and excluded: {n_honeypots}")

        # ---- Stage 4: Generate reasoning and build output DataFrame ----
        logger.info("Stage 4/4: Generating reasoning strings…")
        rows = []
        prev_score = float("inf")

        for rank_pos, r in enumerate(top_records, start=1):
            c      = r["_candidate"]
            scores = r["_scores"]
            final  = r["_final"]

            # Enforce non-increasing scores (handle float precision edge cases)
            final = min(final, prev_score)
            prev_score = final

            reasoning = generate_reasoning(c, scores, rank=rank_pos)

            rows.append({
                "candidate_id": r["candidate_id"],
                "rank":         rank_pos,
                "score":        round(final, 6),
                "reasoning":    reasoning,
            })

        df = pd.DataFrame(rows, columns=["candidate_id", "rank", "score", "reasoning"])

        # ---- Validate non-increasing scores ----------------------------
        scores_arr = df["score"].values
        for i in range(len(scores_arr) - 1):
            if scores_arr[i] < scores_arr[i + 1]:
                logger.warning(
                    f"Score inversion at rank {i+1} → {i+2}: "
                    f"{scores_arr[i]:.6f} < {scores_arr[i+1]:.6f}; clamping"
                )
                scores_arr[i + 1] = scores_arr[i]
        df["score"] = scores_arr

        total_elapsed = time.time() - total_start
        logger.info(f"Pipeline complete in {total_elapsed:.1f}s")
        logger.info(f"Submission shape: {df.shape}  (rows={len(df)}, cols={list(df.columns)})")

        return df

    # ------------------------------------------------------------------
    def score_breakdown(self, candidate: dict[str, Any]) -> dict[str, Any]:
        """
        Diagnostic: compute and return full score breakdown for one candidate.
        Useful for debugging and explaining model decisions.
        """
        sem_score  = self.engine.candidate_similarity(candidate)
        hp_score   = self.honeypot_detector.score(candidate)
        breakdown  = compute_full_score(candidate, sem_score, hp_score)
        breakdown["candidate_id"] = candidate.get("candidate_id", "UNKNOWN")
        return breakdown
