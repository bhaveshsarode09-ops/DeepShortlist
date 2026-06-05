"""
signals.py — DeepShortlist
Processes the 23 redrob_signals per candidate into normalised
availability and engagement sub-scores.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

from src.jd_config import IDEAL_NOTICE_DAYS, PENALTY_NOTICE_DAYS

logger = logging.getLogger(__name__)

_TODAY = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_date(s: Any) -> datetime | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
                "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            dt = datetime.strptime(str(s), fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _days_since(date_str: Any) -> float:
    """Returns days since the given date string; returns 999 if unparsable."""
    dt = _parse_date(date_str)
    if dt is None:
        return 999.0
    return max(0.0, (_TODAY - dt).days)


def _sigmoid(x: float, k: float = 1.0) -> float:
    """Sigmoid used for smooth score transitions."""
    return 1.0 / (1.0 + math.exp(-k * x))


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Availability Score
# Answers: "Can we actually hire this person right now?"
# ---------------------------------------------------------------------------

def compute_availability_score(signals: dict[str, Any]) -> float:
    """
    Combines 8 of the 23 signals most predictive of hire-ability.
    Returns a float in [0, 1].
    """
    score = 0.0

    # 1. Open-to-work flag (hard signal, high weight)
    if signals.get("open_to_work_flag") is True:
        score += 0.25
    elif signals.get("open_to_work_flag") is False:
        score += 0.05   # not zero — they might still respond

    # 2. Last active date (recency decay)
    days_inactive = _days_since(signals.get("last_active_date"))
    if days_inactive <= 3:
        score += 0.25
    elif days_inactive <= 7:
        score += 0.22
    elif days_inactive <= 14:
        score += 0.18
    elif days_inactive <= 30:
        score += 0.14
    elif days_inactive <= 60:
        score += 0.08
    elif days_inactive <= 90:
        score += 0.04
    elif days_inactive <= 180:
        score += 0.01
    # > 180 days → 0 contribution

    # 3. Notice period (<30 days is ideal per JD)
    notice = signals.get("notice_period_days")
    if notice is None:
        score += 0.08   # unknown — neutral
    else:
        notice = float(notice)
        if notice == 0:
            score += 0.20
        elif notice <= IDEAL_NOTICE_DAYS:
            score += 0.20 * (1 - notice / IDEAL_NOTICE_DAYS * 0.3)
        elif notice <= 60:
            score += 0.10
        elif notice <= PENALTY_NOTICE_DAYS:
            score += 0.04
        else:
            score += 0.00  # 90+ day notice → no contribution

    # 4. Recruiter response rate (predictive of engagement)
    rr = signals.get("recruiter_response_rate")
    if rr is not None:
        rr = _clamp(float(rr))
        # Non-linear: low responders are much worse than average
        score += 0.15 * (rr ** 0.7)

    # 5. Average response time (lower = better)
    art = signals.get("avg_response_time_hours")
    if art is not None:
        art = float(art)
        if art <= 2:
            score += 0.08
        elif art <= 12:
            score += 0.06
        elif art <= 24:
            score += 0.04
        elif art <= 72:
            score += 0.02
        # > 72h → 0

    # 6. Interview completion rate
    icr = signals.get("interview_completion_rate")
    if icr is not None:
        icr = _clamp(float(icr))
        score += 0.07 * icr

    # Normalise to [0, 1]
    return _clamp(score)


# ---------------------------------------------------------------------------
# Engagement Score
# Answers: "Is this candidate active, validated by the market, and credible?"
# ---------------------------------------------------------------------------

def compute_engagement_score(signals: dict[str, Any]) -> float:
    """
    Uses the remaining signals to measure platform engagement and
    external technical credibility.
    Returns a float in [0, 1].
    """
    score = 0.0

    # 1. Profile completeness (baseline credibility)
    completeness = signals.get("profile_completeness_score")
    if completeness is not None:
        score += 0.12 * _clamp(float(completeness) / 100)

    # 2. GitHub activity — CRITICAL for AI Engineer role
    github = signals.get("github_activity_score")
    if github is None or github == -1:
        score += 0.00   # No GitHub → negative signal for this role
    else:
        github = _clamp(float(github) / 100)
        score += 0.28 * github  # highest single weight in engagement

    # 3. Saved by recruiters in last 30d (market validation)
    saved = signals.get("saved_by_recruiters_30d")
    if saved is not None:
        saved = _clamp(float(saved) / 20)   # cap at 20 saves
        score += 0.18 * saved

    # 4. Skill assessment scores (Redrob-verified skills)
    assessments = signals.get("skill_assessment_scores") or {}
    if isinstance(assessments, dict) and assessments:
        avg_assess = sum(float(v) for v in assessments.values()) / len(assessments)
        score += 0.18 * _clamp(avg_assess / 100)

    # 5. Endorsements received (peer validation)
    endorsements = signals.get("endorsements_received")
    if endorsements is not None:
        endorse_score = _clamp(float(endorsements) / 50)  # cap at 50
        score += 0.08 * endorse_score

    # 6. Profile views received (30d) — recruiter interest indicator
    views = signals.get("profile_views_received_30d")
    if views is not None:
        view_score = _clamp(float(views) / 30)
        score += 0.06 * view_score

    # 7. LinkedIn connected (trust / verifiability)
    if signals.get("linkedin_connected") is True:
        score += 0.04

    # 8. Verified contact info (reduces ghost-candidate risk)
    verified = 0
    if signals.get("verified_email") is True:
        verified += 1
    if signals.get("verified_phone") is True:
        verified += 1
    score += 0.03 * (verified / 2)

    # 9. Applications submitted in last 30d (active job seeker)
    apps = signals.get("applications_submitted_30d")
    if apps is not None:
        apps_score = _clamp(float(apps) / 10)
        score += 0.03 * apps_score

    return _clamp(score)


# ---------------------------------------------------------------------------
# Offer & interview reliability bonus
# ---------------------------------------------------------------------------

def compute_reliability_bonus(signals: dict[str, Any]) -> float:
    """
    Additional micro-score for candidates who historically accept offers
    and complete interviews — reduces risk of pipeline waste.
    """
    bonus = 0.0

    oar = signals.get("offer_acceptance_rate")
    if oar is not None and oar != -1:
        oar = _clamp(float(oar))
        bonus += 0.5 * oar

    icr = signals.get("interview_completion_rate")
    if icr is not None:
        icr = _clamp(float(icr))
        bonus += 0.5 * icr

    return _clamp(bonus)   # 0-1 bonus multiplier, applied gently in scorer


# ---------------------------------------------------------------------------
# Salary fit
# ---------------------------------------------------------------------------

def salary_in_range(signals: dict[str, Any], band_min: float = 30.0, band_max: float = 70.0) -> bool:
    """
    True if candidate's expected salary overlaps the JD's implied band (INR LPA).
    Redrob Senior AI role in India: estimated 30–70 LPA band.
    """
    salary = signals.get("expected_salary_range_inr_lpa") or {}
    if not isinstance(salary, dict):
        return True   # unknown → don't penalise
    cand_min = salary.get("min") or 0
    cand_max = salary.get("max") or 999
    return not (float(cand_max) < band_min or float(cand_min) > band_max)


# ---------------------------------------------------------------------------
# Connection count (mild social proof)
# ---------------------------------------------------------------------------

def connection_score(signals: dict[str, Any]) -> float:
    cc = signals.get("connection_count")
    if cc is None:
        return 0.5
    return _clamp(float(cc) / 500)   # 500+ connections → full score
