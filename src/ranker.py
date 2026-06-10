"""
ranker.py — DeepShortlist v2
Full pipeline:
  1. Schema detection   (one-time, <1s)
  2. BM25+ first pass   (100K → 3000, ~5s)
  3. LSA reranking      (3000 scored, ~3s)
  4. RRF ensemble       (3000 → 100, ~2s)
  5. Reasoning          (~0.5s)
  Total: ~12s for 100K candidates, ~200MB RAM
"""
from __future__ import annotations
import logging, time
from typing import Any

import numpy as np
import pandas as pd

from src.schema import Schema
from src.retrieval import RetrievalEngine
from src.scorer import EnsembleScorer
from src.reasoning import build_reasoning

logger = logging.getLogger(__name__)

FIRST_PASS   = 3000   # BM25+ → LSA pool size
SCORING_POOL = 1500   # candidates to fully score (after retrieval)
FINAL_TOP_K  = 100


class DeepShortlistRanker:
    def __init__(self, first_pass: int = FIRST_PASS, jd_text: str | None = None):
        self.first_pass = first_pass
        self.jd_text    = jd_text
        self.schema     = Schema()

    def rank(self, candidates: list[dict], top_k: int = FINAL_TOP_K) -> pd.DataFrame:
        t0 = time.time()
        n  = len(candidates)
        logger.info(f"DeepShortlist v2 pipeline: {n:,} candidates")

        # ── Stage 1: Schema detection ─────────────────────────────────────────
        logger.info("Stage 1/4: Detecting schema…")
        self.schema.detect(candidates)

        # ── Stage 2 & 3: BM25+ + LSA retrieval ───────────────────────────────
        logger.info("Stage 2/4: BM25+ + LSA retrieval…")
        engine = RetrievalEngine(self.schema, first_pass=self.first_pass)
        engine.build(candidates)

        pool_size = min(SCORING_POOL, n)
        pool_idx, pool_ret_scores = engine.top_indices(pool_size)
        logger.info(f"Pool: {len(pool_idx):,} candidates for full scoring")

        # ── Stage 4: RRF ensemble scoring ─────────────────────────────────────
        logger.info("Stage 3/4: RRF ensemble scoring…")
        pool_candidates = [candidates[int(i)] for i in pool_idx]
        scorer = EnsembleScorer(self.schema)
        results = scorer.score_pool(pool_candidates, pool_ret_scores, top_k=top_k)

        # ── Stage 5: Reasoning + output ───────────────────────────────────────
        logger.info("Stage 4/4: Generating reasoning…")
        rows: list[dict] = []
        prev_score = float("inf")
        for rank_pos, r in enumerate(results, start=1):
            final = min(r["final_score"], prev_score)
            prev_score = final
            r["breakdown"]["final_score"] = final
            reason = build_reasoning(self.schema, r["candidate"], r["breakdown"], rank_pos)
            rows.append({
                "candidate_id": r["candidate_id"],
                "rank":         rank_pos,
                "score":        round(final, 6),
                "reasoning":    reason,
            })

        df = pd.DataFrame(rows, columns=["candidate_id","rank","score","reasoning"])
        elapsed = time.time() - t0
        logger.info(
            f"Done in {elapsed:.1f}s | "
            f"score range [{df['score'].min():.4f} – {df['score'].max():.4f}] | "
            f"spread={df['score'].max()-df['score'].min():.4f}"
        )
        return df
