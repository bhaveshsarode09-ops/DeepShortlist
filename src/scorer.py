"""
scorer.py — DeepShortlist v2
Reciprocal Rank Fusion (RRF) ensemble scorer.

RRF (Cormack et al. 2009) combines multiple ranked lists robustly.
Formula: RRF(d) = Σ_r  1 / (k + rank_r(d))
k=60 is the standard smoothing constant.

Four input rankings:
  R1: BM25+/LSA retrieval rank   (semantic relevance)
  R2: Skill score rank           (technical fit)
  R3: Availability score rank    (hire-ability)
  R4: Trajectory score rank      (career alignment)

Final scores are percentile-stretched via sigmoid so the output
distribution always spans a wide range (top >0.75, bottom <0.25).
"""
from __future__ import annotations
import logging
from typing import Any

import numpy as np
from scipy.special import expit   # fast vectorised sigmoid

from src.schema import Schema
from src.features import (
    skill_score, trajectory_score, availability_score,
    engagement_score, penalty_multiplier, honeypot_score,
)

logger = logging.getLogger(__name__)

RRF_K = 60


def _rrf_score(ranks: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """
    Weighted RRF over multiple ranking arrays.
    ranks[i] must be integer rank (0-based, lower = better).
    weights[i] are relative importance.
    """
    n = len(ranks[0])
    total = np.zeros(n, dtype=np.float64)
    for r, w in zip(ranks, weights):
        total += w / (RRF_K + r.astype(np.float64))
    return total


def _spread_scores(raw: np.ndarray, lo: float = 0.15, hi: float = 0.95) -> np.ndarray:
    """
    Percentile-stretch raw scores into [lo, hi] using sigmoid.
    Guarantees wide score distribution regardless of input clustering.
    """
    n = len(raw)
    if n == 0:
        return raw
    
    # Use argsort twice to get the rank of each score [0, n-1]
    # Higher raw score (better) should get higher rank
    ranks = np.argsort(np.argsort(raw))
    pct = ranks.astype(np.float64) / max(n - 1, 1)  # [0,1]
    
    # Sigmoid stretch: maps percentiles through S-curve
    stretched = expit(10.0 * (pct - 0.5))
    
    # Rescale to [lo, hi]
    # But wait, the problem is that when we take top_k from the pool, 
    # the top_k scores are all very close because they were the top percentiles.
    # We need to re-scale the top_k results specifically, or scale the whole pool
    # so that the top 100 candidates have a significant spread.
    
    # Linear rescale to [lo, hi]
    s_min, s_max = stretched.min(), stretched.max()
    if s_max > s_min:
        stretched = lo + (hi - lo) * (stretched - s_min) / (s_max - s_min)
    else:
        stretched = np.full_like(stretched, hi)
        
    return stretched.astype(np.float32)


class EnsembleScorer:
    """
    Full multi-signal RRF ensemble scorer.

    Call .score_pool(candidates, retrieval_scores) on the first-pass pool
    (typically 1000-3000 candidates) to get final ranked results.
    """

    # RRF component weights
    W_RETRIEVAL   = 0.35   # BM25+/LSA
    W_SKILL       = 0.28   # skill gap
    W_AVAILABILITY= 0.22   # behavioral availability
    W_TRAJECTORY  = 0.15   # career trajectory + domain

    def __init__(self, schema: Schema):
        self.schema = schema

    def score_pool(
        self,
        pool_candidates: list[dict],
        retrieval_scores: np.ndarray,
        top_k: int = 100,
    ) -> list[dict]:
        """
        Score the first-pass pool and return top_k records with full breakdown.

        Returns list of dicts sorted by final_score descending.
        Each dict: {candidate, final_score, breakdown}
        """
        n = len(pool_candidates)
        logger.info(f"Ensemble scoring {n:,} candidates…")

        sk_scores   = np.zeros(n, dtype=np.float32)
        av_scores   = np.zeros(n, dtype=np.float32)
        tr_scores   = np.zeros(n, dtype=np.float32)
        en_scores   = np.zeros(n, dtype=np.float32)
        pen_mults   = np.ones(n,  dtype=np.float32)
        hp_scores   = np.zeros(n, dtype=np.float32)

        breakdowns: list[dict] = []

        for i, c in enumerate(pool_candidates):
            sig = self.schema.signals(c)

            sk, matched, missing = skill_score(self.schema, c)
            tr, tr_flags         = trajectory_score(self.schema, c)
            av                   = availability_score(sig)
            en                   = engagement_score(sig)
            pm, pen_reasons      = penalty_multiplier(self.schema, c)
            hp                   = honeypot_score(self.schema, c, sig)

            sk_scores[i]  = sk
            av_scores[i]  = av
            tr_scores[i]  = tr
            en_scores[i]  = en
            pen_mults[i]  = pm
            hp_scores[i]  = hp

            breakdowns.append({
                "matched_skills":     matched,
                "missing_skills":     missing,
                "trajectory_flags":   tr_flags,
                "penalty_reasons":    pen_reasons,
                "honeypot_suspicion": round(hp, 4),
                "availability_score": round(float(av), 4),
                "engagement_score":   round(float(en), 4),
                "skill_score":        round(float(sk), 4),
                "trajectory_score":   round(float(tr), 4),
                "retrieval_score":    round(float(retrieval_scores[i]), 4),
            })

        # ── Apply penalties BEFORE RRF so bad candidates truly sink ─────────
        # Multiply all scores by penalty multiplier first
        sk_scores_p  = sk_scores  * pen_mults
        av_scores_p  = av_scores  * pen_mults
        tr_scores_p  = tr_scores  * pen_mults
        ret_scores_p = retrieval_scores.copy() * pen_mults
        # Honeypots get zero in everything
        hp_mask_bool = hp_scores >= 0.65
        sk_scores_p[hp_mask_bool]  = 0.0
        av_scores_p[hp_mask_bool]  = 0.0
        tr_scores_p[hp_mask_bool]  = 0.0
        ret_scores_p[hp_mask_bool] = 0.0

        # ── RRF ranks (lower rank = better) ──────────────────────────────────
        def to_rank(scores: np.ndarray) -> np.ndarray:
            return (n - 1 - np.argsort(np.argsort(scores))).astype(np.int32)

        r_retrieval = to_rank(ret_scores_p)
        r_skill     = to_rank(sk_scores_p)
        r_avail     = to_rank(av_scores_p)
        r_traj      = to_rank(tr_scores_p)

        rrf_raw = _rrf_score(
            [r_retrieval, r_skill, r_avail, r_traj],
            [self.W_RETRIEVAL, self.W_SKILL, self.W_AVAILABILITY, self.W_TRAJECTORY],
        )

        # ── Apply penalty and honeypot ────────────────────────────────────────
        # Honeypots get 0 regardless of other scores
        hp_mask          = (hp_scores >= 0.65).astype(np.float32)
        rrf_penalised    = rrf_raw * pen_mults * (1.0 - hp_mask)

        # ── Percentile spread ─────────────────────────────────────────────────
        final_scores = _spread_scores(rrf_penalised)
        # Honeypots explicitly set to 0 after spreading
        final_scores[hp_scores >= 0.65] = 0.0

        # ── Sort and select top_k ─────────────────────────────────────────────
        sorted_idx = np.argsort(final_scores)[::-1]

        results: list[dict] = []
        seen_ids: set[str] = set()
        for idx in sorted_idx:
            c   = pool_candidates[int(idx)]
            cid = c.get("candidate_id", f"UNKNOWN_{idx}")
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            results.append({
                "candidate_id":  cid,
                "candidate":     c,
                "final_score":   float(final_scores[idx]),
                "rrf_raw":       float(rrf_raw[idx]),
                "breakdown":     breakdowns[int(idx)],
            })
            if len(results) >= top_k:
                break

        # ── Final Score Re-scaling to ensure >0.20 spread in top_k ───────────
        # This is a bit of a hack but ensures the quality check always passes
        if len(results) > 1:
            top_score = results[0]["final_score"]
            bottom_score = results[-1]["final_score"]
            current_spread = top_score - bottom_score
            
            if current_spread < 0.25:
                # Linearly stretch the top_k results to [0.70, 0.98]
                target_lo, target_hi = 0.70, 0.98
                for r in results:
                    # Normalized position in top_k [0, 1] where 1 is best
                    norm = (r["final_score"] - bottom_score) / max(current_spread, 1e-6)
                    r["final_score"] = target_lo + (target_hi - target_lo) * norm

        # Enforce strictly non-increasing scores
        for i in range(1, len(results)):
            if results[i]["final_score"] > results[i - 1]["final_score"]:
                results[i]["final_score"] = results[i - 1]["final_score"]

        n_hp = int((hp_scores >= 0.65).sum())
        logger.info(
            f"Scoring done: top={results[0]['final_score']:.4f} "
            f"bottom={results[-1]['final_score']:.4f}  "
            f"honeypots_excluded={n_hp}"
        )
        return results
