"""
scorer.py — DeepShortlist
Multi-signal weighted scorer that combines semantic, skill, trajectory,
availability, engagement, and location into a final 0-1 score.
"""

from __future__ import annotations

import logging
from typing import Any

from src.jd_config import (
    IDEAL_MAX_EXP,
    IDEAL_MIN_EXP,
    HARD_MIN_EXP,
    HARD_MAX_EXP,
    REQUIRED_SKILLS,
    PREFERRED_SKILLS,
    DISQUALIFIER_TITLES,
    PURE_CONSULTING_COMPANIES,
    DISQUALIFIER_DOMAINS,
    RESEARCH_ONLY_INDICATORS,
    TARGET_LOCATIONS,
    WEIGHTS,
    PENALTIES,
)
from src.signals import (
    compute_availability_score,
    compute_engagement_score,
    compute_reliability_bonus,
)
from src.field_extractor import (
    get_title, get_yoe, get_location, get_skills,
    get_work_history, get_company_names, get_job_titles,
    get_signals, build_rich_text,
)

logger = logging.getLogger(__name__)


def compute_full_score(
    c: dict[str, Any],
    semantic_score: float,
    honeypot_suspicion: float = 0.0,
) -> dict[str, Any]:
    """
    Computes the final composite score and returns a breakdown for reasoning.
    """
    signals = get_signals(c)
    
    # 1. Semantic Score (already computed via FAISS/Engine)
    # Weight: 30%
    
    # 2. Skill Gap Score (20%)
    skill_score, matched, missing = _compute_skill_score(c)
    
    # 3. Trajectory Score (15%)
    trajectory_score, traj_flags = _compute_trajectory_score(c)
    
    # 4. Availability Score (20%)
    availability_score = compute_availability_score(signals)
    
    # 5. Engagement Score (10%)
    engagement_score = compute_engagement_score(signals)
    
    # 6. Location Score (5%)
    location_score = _compute_location_score(c)
    
    # Weighted Sum
    raw_score = (
        semantic_score     * WEIGHTS["semantic"] +
        skill_score        * WEIGHTS["skill_gap"] +
        trajectory_score   * WEIGHTS["trajectory"] +
        availability_score * WEIGHTS["availability"] +
        engagement_score   * WEIGHTS["engagement"] +
        location_score     * WEIGHTS["location"]
    )
    
    # Apply Reliability Bonus (gently)
    reliability = compute_reliability_bonus(signals)
    raw_score = raw_score * (1.0 + 0.05 * reliability)
    
    # Apply Penalty Multipliers
    final_score = raw_score
    penalty_reasons = []
    
    # Honeypot
    if honeypot_suspicion >= 0.60:
        final_score *= PENALTIES["honeypot"]
        penalty_reasons.append("honeypot")
    
    # Wrong Title
    title = get_title(c).lower()
    if any(dt in title for dt in DISQUALIFIER_TITLES):
        final_score *= PENALTIES["wrong_title"]
        penalty_reasons.append("wrong job family")
        
    # Consulting Only
    work_history = get_work_history(c)
    if work_history:
        cos = [str(j.get("company", "")).lower() for j in work_history if isinstance(j, dict)]
        if cos and all(any(pc in co for pc in PURE_CONSULTING_COMPANIES) for co in cos):
            final_score *= PENALTIES["consulting_only"]
            penalty_reasons.append("consulting background")
            
    # Research Only
    if any(ind in title for ind in RESEARCH_ONLY_INDICATORS):
        final_score *= PENALTIES["research_only"]
        penalty_reasons.append("research focus")
        
    # Experience Too Low
    yoe = get_yoe(c)
    if yoe is not None and yoe < HARD_MIN_EXP and yoe > 0:
        final_score *= PENALTIES["experience_too_low"]
        penalty_reasons.append("low experience")

    return {
        "final_score":        float(final_score),
        "semantic_score":     float(semantic_score),
        "skill_score":        float(skill_score),
        "trajectory_score":   float(trajectory_score),
        "availability_score": float(availability_score),
        "engagement_score":   float(engagement_score),
        "location_score":     float(location_score),
        "matched_skills":     matched,
        "missing_skills":     missing,
        "trajectory_flags":   traj_flags,
        "penalty_reasons":    penalty_reasons,
        "honeypot_suspicion": float(honeypot_suspicion),
    }


def _compute_skill_score(c: dict[str, Any]) -> tuple[float, list[str], list[str]]:
    cand_skills = [str(s).lower() for s in get_skills(c)]
    matched_req = [s for s in REQUIRED_SKILLS if any(s in cs for cs in cand_skills)]
    matched_pref = [s for s in PREFERRED_SKILLS if any(s in cs for cs in cand_skills)]
    
    missing_req = [s for s in REQUIRED_SKILLS if s not in matched_req]
    
    # Score: 0.7 for required, 0.3 for preferred
    req_ratio = len(matched_req) / len(REQUIRED_SKILLS) if REQUIRED_SKILLS else 1.0
    pref_ratio = min(1.0, len(matched_pref) / 10) if PREFERRED_SKILLS else 1.0
    
    score = (0.7 * req_ratio) + (0.3 * pref_ratio)
    return score, matched_req, missing_req


def _compute_trajectory_score(c: dict[str, Any]) -> tuple[float, list[str]]:
    score = 0.0  # reset to 0 to handle new logic correctly
    flags = []
    
    # YOE Fit
    yoe = get_yoe(c)
    if yoe is None:
        score += 0.20
        flags.append("experience unknown")
    elif yoe < HARD_MIN_EXP:
        score += 0.10
        flags.append(f"under-experienced ({yoe:.1f}y)")
    elif IDEAL_MIN_EXP <= yoe <= IDEAL_MAX_EXP:
        score += 0.40
    elif HARD_MIN_EXP <= yoe < IDEAL_MIN_EXP:
        score += 0.25
    elif IDEAL_MAX_EXP < yoe <= 12:
        score += 0.35
    else:
        score += 0.25
        flags.append(f"potentially overqualified ({yoe:.1f}y)")
        
    # Domain check
    bio = (c.get("bio") or "").lower()
    if any(d in bio for d in DISQUALIFIER_DOMAINS):
        score -= 0.2
        flags.append("domain mismatch (CV/Robotics/Speech)")
        
    return max(0.0, min(1.0, score)), flags


def _compute_location_score(c: dict[str, Any]) -> float:
    loc = get_location(c).lower()
    if any(tl in loc for tl in TARGET_LOCATIONS):
        return 1.0
    if get_signals(c).get("willing_to_relocate") is True:
        return 0.7
    return 0.2
