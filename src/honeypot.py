"""
honeypot.py — DeepShortlist
Detects the ~80 synthetic honeypot candidates in the dataset
whose profiles are subtly impossible or internally inconsistent.
Honeypots receive a score multiplier of 0.0 (effectively rank 101+).

Detection strategy (layered):
  Rule 1: Experience > company lifespan
  Rule 2: Too many "expert"-level skills with implausible coverage
  Rule 3: Signal envelope violations (values outside declared ranges)
  Rule 4: Temporal impossibility (signup before graduation, etc.)
  Rule 5: Statistical outlier across multiple dimensions simultaneously
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_TODAY = datetime.now(timezone.utc)

# Known founding years of companies that appear in synthetic data
# (extend this if you spot more patterns in the dataset)
_COMPANY_FOUNDED: dict[str, int] = {
    "openai":          2015,
    "anthropic":       2021,
    "mistral":         2023,
    "cohere":          2019,
    "hugging face":    2016,
    "huggingface":     2016,
    "stability ai":    2020,
    "redrob":          2021,
    "perplexity":      2022,
}

# Skills that are mutually exclusive or impossible to be expert in simultaneously
# without extraordinary time investment
_SKILL_GROUP_SIZE_LIMIT: dict[str, int] = {
    "deep learning frameworks": 4,    # pytorch, tensorflow, jax, paddle
    "vector databases":         6,    # faiss, pinecone, weaviate, qdrant, milvus, chroma
    "cloud platforms":          3,    # AWS, GCP, Azure
    "programming languages":    8,    # py, go, rust, scala, java, c++, r, julia
}

# Maximum plausible years of experience for realistic profiles
_MAX_PLAUSIBLE_EXP = 30.0
_MIN_PLAUSIBLE_EXP = 0.0


def _parse_year(s: Any) -> int | None:
    if not s:
        return None
    m = re.search(r"(\d{4})", str(s))
    return int(m.group(1)) if m else None


def _days_since(date_str: Any) -> float:
    if not date_str:
        return 0.0
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            dt = datetime.strptime(str(date_str), fmt).replace(tzinfo=timezone.utc)
            return max(0.0, (_TODAY - dt).days)
        except ValueError:
            pass
    return 0.0


class HoneypotDetector:
    """
    Multi-rule layered honeypot detection.
    Call .is_honeypot(candidate) for a boolean decision and
    .score(candidate) for a continuous suspicion score in [0, 1].
    """

    def __init__(self, suspicion_threshold: float = 0.60) -> None:
        self.threshold = suspicion_threshold
        self._flagged: set[str] = set()

    # ------------------------------------------------------------------
    def score(self, c: dict[str, Any]) -> float:
        """Returns suspicion score: 0.0 = clean, 1.0 = certain honeypot."""
        suspicion = 0.0
        reasons: list[str] = []

        cid = c.get("candidate_id", "UNKNOWN")

        # --- Rule 1: Experience vs company lifespan ---
        yoe = float(c.get("years_of_experience") or c.get("years_experience") or 0)
        suspicion += self._rule_experience_vs_company(c, yoe, reasons)

        # --- Rule 2: Implausible total experience ---
        if yoe > _MAX_PLAUSIBLE_EXP:
            suspicion += 0.40
            reasons.append(f"experience={yoe}y exceeds plausible max={_MAX_PLAUSIBLE_EXP}y")
        elif yoe < _MIN_PLAUSIBLE_EXP:
            suspicion += 0.20
            reasons.append("negative experience")

        # --- Rule 3: Too many skills with perfect assessment scores ---
        suspicion += self._rule_skill_explosion(c, reasons)

        # --- Rule 4: Signal envelope violations ---
        suspicion += self._rule_signal_envelope(c, reasons)

        # --- Rule 5: Temporal impossibility ---
        suspicion += self._rule_temporal(c, yoe, reasons)

        # --- Rule 6: Perfect-across-the-board anomaly ---
        suspicion += self._rule_too_perfect(c, reasons)

        if suspicion >= self.threshold:
            self._flagged.add(cid)
            if reasons:
                logger.debug(f"[HONEYPOT] {cid}: {'; '.join(reasons[:3])}")

        return min(suspicion, 1.0)

    # ------------------------------------------------------------------
    def is_honeypot(self, c: dict[str, Any]) -> bool:
        return self.score(c) >= self.threshold

    @property
    def flagged_ids(self) -> frozenset[str]:
        return frozenset(self._flagged)

    # ------------------------------------------------------------------
    # Individual rules
    # ------------------------------------------------------------------

    def _rule_experience_vs_company(
        self, c: dict[str, Any], yoe: float, reasons: list[str]
    ) -> float:
        """Detects 8-years-at-company-founded-3-years-ago style honeypots."""
        work_history = c.get("work_history") or c.get("experience") or []
        suspicion = 0.0
        for job in work_history:
            if not isinstance(job, dict):
                continue
            company_name = (job.get("company") or "").lower()
            duration = float(job.get("duration_years") or job.get("years") or 0)
            if duration <= 0:
                continue
            for known_co, founded in _COMPANY_FOUNDED.items():
                if known_co in company_name:
                    max_tenure = max(0, _TODAY.year - founded)
                    if duration > max_tenure + 0.5:   # +0.5 year tolerance
                        suspicion += 0.55
                        reasons.append(
                            f"{company_name} founded {founded}, "
                            f"but tenure={duration:.1f}y (impossible)"
                        )
        return min(suspicion, 0.60)

    def _rule_skill_explosion(self, c: dict[str, Any], reasons: list[str]) -> float:
        """Detects suspiciously large skill lists with perfect assessments."""
        skills = c.get("skills") or []
        if not isinstance(skills, list):
            return 0.0

        suspicion = 0.0
        if len(skills) > 40:
            suspicion += 0.25
            reasons.append(f"{len(skills)} skills listed (suspiciously many)")

        # All assessment scores perfect (100)?
        assessments = (c.get("redrob_signals") or {}).get("skill_assessment_scores") or {}
        if isinstance(assessments, dict) and len(assessments) >= 5:
            perfect = sum(1 for v in assessments.values() if float(v) >= 98)
            if perfect == len(assessments) and len(assessments) >= 5:
                suspicion += 0.35
                reasons.append(f"All {len(assessments)} assessments are ≥98/100")
            elif perfect / len(assessments) > 0.80:
                suspicion += 0.15
                reasons.append(f"{perfect}/{len(assessments)} assessments ≥98/100")

        return min(suspicion, 0.50)

    def _rule_signal_envelope(self, c: dict[str, Any], reasons: list[str]) -> float:
        """Check signals are within declared ranges (0-100, 0.0-1.0, etc.)."""
        signals = c.get("redrob_signals") or {}
        suspicion = 0.0

        checks = [
            ("profile_completeness_score", 0, 100),
            ("recruiter_response_rate",    0.0, 1.0),
            ("interview_completion_rate",  0.0, 1.0),
            ("offer_acceptance_rate",     -1.0, 1.0),
            ("github_activity_score",     -1, 100),
            ("notice_period_days",          0, 180),
        ]

        for key, lo, hi in checks:
            val = signals.get(key)
            if val is None:
                continue
            try:
                fval = float(val)
                if fval < lo or fval > hi:
                    suspicion += 0.15
                    reasons.append(f"{key}={fval} outside [{lo},{hi}]")
            except (TypeError, ValueError):
                pass

        return min(suspicion, 0.40)

    def _rule_temporal(self, c: dict[str, Any], yoe: float, reasons: list[str]) -> float:
        """Signup date vs claimed experience sanity check."""
        signals   = c.get("redrob_signals") or {}
        signup_ds = signals.get("signup_date")
        if not signup_ds:
            return 0.0

        days_since_signup = _days_since(signup_ds)
        years_since_signup = days_since_signup / 365.25

        # If yoe >> years since signup (impossible to have accumulated that exp)
        suspicion = 0.0
        if yoe > 0 and years_since_signup < 0.1:
            # just signed up but has 7 years exp → that's fine
            pass
        # Check if signup year is after graduation (rough)
        edu = c.get("education") or []
        for e in edu:
            if not isinstance(e, dict):
                continue
            grad_year = _parse_year(e.get("graduation_year") or e.get("year"))
            signup_year = _parse_year(signup_ds)
            if grad_year and signup_year and signup_year < grad_year:
                suspicion += 0.30
                reasons.append(f"Signup {signup_year} before graduation {grad_year}")

        return min(suspicion, 0.35)

    def _rule_too_perfect(self, c: dict[str, Any], reasons: list[str]) -> float:
        """
        A candidate who is simultaneously in the 99th percentile across
        EVERY dimension is statistically implausible.
        """
        signals   = c.get("redrob_signals") or {}
        high_flags = 0

        thresholds = [
            ("profile_completeness_score", 99),
            ("recruiter_response_rate",    0.98),
            ("interview_completion_rate",  0.98),
            ("github_activity_score",      99),
            ("saved_by_recruiters_30d",    50),
            ("endorsements_received",      200),
        ]
        for key, threshold in thresholds:
            val = signals.get(key)
            if val is not None:
                try:
                    if float(val) >= threshold:
                        high_flags += 1
                except (TypeError, ValueError):
                    pass

        if high_flags >= 5:
            reasons.append(f"Outlier across {high_flags} dimensions simultaneously")
            return 0.35
        if high_flags == 4:
            return 0.15
        return 0.0
