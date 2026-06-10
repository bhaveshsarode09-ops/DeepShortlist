"""
features.py — DeepShortlist v2
Extracts 4 feature scores from each candidate:
  1. SkillScore   — required/preferred skill coverage
  2. TrajectScore — career trajectory + domain alignment
  3. AvailScore   — availability from 23 behavioral signals
  4. EngageScore  — platform engagement + GitHub activity

All scores normalised to [0, 1].
"""
from __future__ import annotations
import logging, math, re
from datetime import datetime, timezone
from typing import Any

import numpy as np

from src.schema import (
    Schema, REQUIRED_SKILLS, PREFERRED_SKILLS,
    DISQUALIFIER_TITLES, CONSULTING_FIRMS, DOMAIN_BOOST_TERMS,
    DOMAIN_PENALTY_TERMS, TARGET_LOCATIONS,
    IDEAL_EXP_MIN, IDEAL_EXP_MAX, HARD_EXP_MIN,
)

logger = logging.getLogger(__name__)
_TODAY = datetime.now(timezone.utc)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clamp(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def _sigmoid(x: float, k: float = 5.0) -> float:
    return 1.0 / (1.0 + math.exp(-k * x))

def _days_since(s: Any) -> float:
    if not s: return 999.0
    for fmt in ("%Y-%m-%d","%Y-%m-%dT%H:%M:%S","%Y-%m-%dT%H:%M:%SZ",
                "%d-%m-%Y","%d/%m/%Y","%Y/%m/%d"):
        try:
            dt = datetime.strptime(str(s), fmt).replace(tzinfo=timezone.utc)
            return max(0.0, (_TODAY - dt).days)
        except ValueError:
            pass
    return 999.0


# ── 1. Skill Score ────────────────────────────────────────────────────────────

def skill_score(schema: Schema, c: dict) -> tuple[float, list[str], list[str]]:
    """
    Fraction of required + preferred skills present in candidate.
    Returns (score, matched_skills, missing_req_skills).
    """
    candidate_skills = schema.skills(c)
    full_text = schema.build_text(c).lower()

    matched_req, missing_req, matched_pref = [], [], []

    for req in REQUIRED_SKILLS:
        hit = (any(req in s for s in candidate_skills) or req in full_text)
        (matched_req if hit else missing_req).append(req)

    for pref in PREFERRED_SKILLS:
        if any(pref in s for s in candidate_skills) or pref in full_text:
            matched_pref.append(pref)

    # Weighted: required contributes 80%, preferred 20%
    req_frac  = len(matched_req) / max(len(REQUIRED_SKILLS), 1)
    pref_frac = len(matched_pref) / max(len(PREFERRED_SKILLS), 1)
    score = _clamp(0.80 * req_frac + 0.20 * pref_frac)

    matched = list(dict.fromkeys(matched_req + matched_pref))[:10]
    return score, matched, missing_req[:8]


# ── 2. Trajectory Score ───────────────────────────────────────────────────────

def trajectory_score(schema: Schema, c: dict) -> tuple[float, list[str]]:
    """
    Evaluates career direction, domain alignment, and experience bracket.
    """
    flags: list[str] = []
    score = 0.0
    text  = schema.build_text(c).lower()
    yoe   = schema.exp(c)
    companies = schema.company_names(c)

    # ── Experience bracket ────────────────────────────────────────────────────
    if yoe is None:
        score += 0.15   # unknown → slight penalty, not zero
        flags.append("experience unknown")
    elif yoe < HARD_EXP_MIN:
        score += 0.05
        flags.append(f"under-experienced ({yoe:.1f}y)")
    elif IDEAL_EXP_MIN <= yoe <= IDEAL_EXP_MAX:
        score += 0.40   # ideal sweet spot
    elif HARD_EXP_MIN <= yoe < IDEAL_EXP_MIN:
        score += 0.20
    elif IDEAL_EXP_MAX < yoe <= 12:
        score += 0.35
    else:
        score += 0.25
        flags.append(f"overqualified ({yoe:.1f}y)")

    # ── Domain alignment ──────────────────────────────────────────────────────
    boost_hits   = sum(1 for kw in DOMAIN_BOOST_TERMS if kw in text)
    penalty_hits = sum(1 for kw in DOMAIN_PENALTY_TERMS if kw in text)
    nlp_ir_hits  = sum(1 for kw in ["nlp","retrieval","embedding","semantic","ranking","search"] if kw in text)

    domain_score = boost_hits / max(len(DOMAIN_BOOST_TERMS), 1)
    if penalty_hits > 0 and nlp_ir_hits == 0:
        domain_score *= 0.3   # CV/speech primary, no NLP → penalise
        flags.append("wrong domain (CV/speech primary)")

    score += domain_score * 0.35

    # ── Product company (not pure consulting) ─────────────────────────────────
    is_only_consulting = (
        len(companies) > 0 and
        all(any(cf in co for cf in CONSULTING_FIRMS) for co in companies)
    )
    if not is_only_consulting:
        score += 0.15
    else:
        flags.append("consulting-only background")

    # ── Seniority signal ──────────────────────────────────────────────────────
    if yoe and yoe >= 4 and any(kw in text for kw in ["senior","lead","principal","staff","head","architect"]):
        score += 0.10

    return _clamp(score), flags


# ── 3. Availability Score (from 23 behavioral signals) ───────────────────────

def availability_score(signals: dict) -> float:
    """
    Combines 8 high-signal behavioral features.
    Answers: "Can we actually hire this person now?"
    """
    s = 0.0

    # Open-to-work (strongest single signal)
    otw = signals.get("open_to_work_flag")
    s += 0.28 if otw is True else (0.05 if otw is False else 0.12)

    # Recency of activity (exponential decay)
    days = _days_since(signals.get("last_active_date"))
    if   days <=  3: s += 0.24
    elif days <=  7: s += 0.21
    elif days <= 14: s += 0.17
    elif days <= 30: s += 0.12
    elif days <= 60: s += 0.06
    elif days <= 90: s += 0.02

    # Notice period
    notice = signals.get("notice_period_days")
    if notice is not None:
        n = float(notice)
        if   n ==  0: s += 0.20
        elif n <= 15: s += 0.18
        elif n <= 30: s += 0.15
        elif n <= 60: s += 0.08
        elif n <= 90: s += 0.02

    # Recruiter response rate (non-linear)
    rr = signals.get("recruiter_response_rate")
    if rr is not None:
        s += 0.15 * (_clamp(float(rr)) ** 0.6)

    # Avg response time
    art = signals.get("avg_response_time_hours")
    if art is not None:
        art = float(art)
        if   art <=  2: s += 0.08
        elif art <= 12: s += 0.06
        elif art <= 24: s += 0.04
        elif art <= 72: s += 0.01

    # Interview completion
    icr = signals.get("interview_completion_rate")
    if icr is not None:
        s += 0.05 * _clamp(float(icr))

    return _clamp(s)


# ── 4. Engagement Score ───────────────────────────────────────────────────────

def engagement_score(signals: dict) -> float:
    """
    Platform engagement + external technical credibility.
    GitHub activity weighted heavily for AI Engineer role.
    """
    s = 0.0

    # GitHub activity (critical for AI Engineer)
    gh = signals.get("github_activity_score")
    if gh is not None and gh != -1:
        s += 0.30 * _clamp(float(gh) / 100)
    # No GitHub → 0 contribution (negative signal, not penalised twice)

    # Profile completeness
    pc = signals.get("profile_completeness_score")
    if pc is not None:
        s += 0.12 * _clamp(float(pc) / 100)

    # Saved by recruiters (market validation — other recruiters also want them)
    saved = signals.get("saved_by_recruiters_30d")
    if saved is not None:
        s += 0.18 * _clamp(float(saved) / 25)

    # Skill assessments (Redrob-verified scores)
    assessments = signals.get("skill_assessment_scores") or {}
    if isinstance(assessments, dict) and assessments:
        avg_a = sum(float(v) for v in assessments.values()) / len(assessments)
        s += 0.18 * _clamp(avg_a / 100)

    # Endorsements
    end = signals.get("endorsements_received")
    if end is not None:
        s += 0.08 * _clamp(float(end) / 60)

    # Profile views (recruiter interest)
    pv = signals.get("profile_views_received_30d")
    if pv is not None:
        s += 0.06 * _clamp(float(pv) / 35)

    # Trust signals
    trust = int(signals.get("verified_email", False) is True) + \
            int(signals.get("verified_phone", False) is True) + \
            int(signals.get("linkedin_connected", False) is True)
    s += 0.04 * (trust / 3)

    # Offer acceptance rate (reliability)
    oar = signals.get("offer_acceptance_rate")
    if oar is not None and oar != -1:
        s += 0.04 * _clamp(float(oar))

    return _clamp(s)


# ── Penalty detection ─────────────────────────────────────────────────────────

def penalty_multiplier(schema: Schema, c: dict) -> tuple[float, list[str]]:
    """
    Returns (multiplier, reasons).
    1.0 = no penalty. Values < 1 reduce final score.
    """
    mult    = 1.0
    reasons = []
    text    = schema.build_text(c).lower()
    title   = schema.title(c).lower()
    yoe     = schema.exp(c)
    cos     = schema.company_names(c)

    # Wrong job family
    for dt in DISQUALIFIER_TITLES:
        if dt in title:
            mult = min(mult, 0.35)
            reasons.append(f"wrong title: {schema.title(c)!r}")
            break

    # Pure consulting background
    if cos and all(any(cf in co for cf in CONSULTING_FIRMS) for co in cos):
        mult = min(mult, 0.55)
        reasons.append("consulting-only background")

    # Wrong primary domain (CV/speech without any NLP/IR)
    has_nlp = any(kw in text for kw in ["nlp","retrieval","embedding","search","ranking"])
    if not has_nlp and any(kw in text for kw in ["computer vision","speech recognition","robotics","ocr"]):
        mult = min(mult, 0.60)
        reasons.append("wrong domain: CV/speech primary, no NLP/IR")

    # Too little experience
    if yoe is not None and yoe < HARD_EXP_MIN:
        mult = min(mult, 0.70)
        reasons.append(f"insufficient experience ({yoe:.1f}y)")

    return mult, reasons


# ── Honeypot detection ────────────────────────────────────────────────────────

def honeypot_score(schema: Schema, c: dict, signals: dict) -> float:
    """
    Returns suspicion score [0,1]. ≥ 0.65 = honeypot.
    Uses 5 independent rules; any 2+ firing = flagged.
    """
    suspicion = 0.0
    FOUNDING = {"openai":2015,"anthropic":2021,"mistral":2023,"cohere":2019,
                "perplexity":2022,"redrob":2021,"stability ai":2020}

    # Rule 1: experience > 35 years
    yoe = schema.exp(c)
    if yoe is not None and yoe > 35:
        suspicion += 0.50

    # Rule 2: tenure > company lifespan
    for job in schema.work_history(c):
        co  = str(job.get("company") or "").lower()
        dur = 0.0
        for k in ["duration_years","years","tenure_years","duration"]:
            try: dur = float(job.get(k, 0) or 0); break
            except: pass
        for name, yr in FOUNDING.items():
            if name in co and dur > (2026 - yr + 0.5):
                suspicion += 0.55

    # Rule 3: too many skills (>45) or all assessments ≥ 98
    if len(schema.skills(c)) > 45:
        suspicion += 0.25
    assessments = signals.get("skill_assessment_scores") or {}
    if isinstance(assessments, dict) and len(assessments) >= 5:
        if sum(1 for v in assessments.values() if float(v) >= 98) == len(assessments):
            suspicion += 0.35

    # Rule 4: signal envelope violations
    for key, lo, hi in [("profile_completeness_score",0,100),
                         ("recruiter_response_rate",0,1),
                         ("interview_completion_rate",0,1),
                         ("github_activity_score",-1,100)]:
        val = signals.get(key)
        if val is not None:
            try:
                fv = float(val)
                if fv < lo or fv > hi:
                    suspicion += 0.15
            except: pass

    # Rule 5: too-perfect across 5+ dimensions simultaneously
    high = sum(1 for key, thresh in [
        ("profile_completeness_score",99),("recruiter_response_rate",0.97),
        ("interview_completion_rate",0.97),("github_activity_score",98),
        ("saved_by_recruiters_30d",80),("endorsements_received",300),
    ] if signals.get(key) is not None and float(signals[key]) >= thresh)
    if high >= 4:
        suspicion += 0.35

    return min(suspicion, 1.0)
