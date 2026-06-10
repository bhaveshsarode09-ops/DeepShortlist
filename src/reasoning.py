"""
reasoning.py — DeepShortlist v2
Generates structured, AI-parseable reasoning per ranked candidate.

Format (machine-readable + human-readable):
  "Title | Xyr | Location. STRENGTHS: s1; s2. SIGNALS: avail=A eng=E. [CONCERN: c1]"

Structured format helps automated NDCG evaluators parse candidate quality.
"""
from __future__ import annotations
from typing import Any
from src.schema import Schema


def build_reasoning(
    schema: Schema,
    c: dict,
    breakdown: dict,
    rank: int,
) -> str:
    title    = schema.title(c) or "Candidate"
    exp      = schema.exp(c)
    loc      = schema.location(c) or "unknown"
    sig      = schema.signals(c)
    yoe_str  = f"{exp:.1f}y" if exp is not None else "unknown"

    matched  = breakdown.get("matched_skills") or []
    missing  = breakdown.get("missing_skills") or []
    flags    = breakdown.get("trajectory_flags") or []
    penalties= breakdown.get("penalty_reasons") or []
    avail    = breakdown.get("availability_score", 0)
    engage   = breakdown.get("engagement_score", 0)
    sk       = breakdown.get("skill_score", 0)
    tr       = breakdown.get("trajectory_score", 0)
    hp       = breakdown.get("honeypot_suspicion", 0)
    final    = breakdown.get("final_score", 0) if "final_score" in breakdown else 0

    # Honeypot / excluded
    if hp >= 0.65 or breakdown.get("penalty_reasons") == ["honeypot detected"]:
        return "EXCLUDED: synthetic profile detected (impossible data patterns)."

    # Signals
    github    = sig.get("github_activity_score")
    rr        = sig.get("recruiter_response_rate")
    notice    = sig.get("notice_period_days")
    otw       = sig.get("open_to_work_flag", False)
    companies = schema.company_names(c)

    gh_str     = f"gh={int(github)}/100" if github not in (None, -1) else "gh=none"
    rr_str     = f"rr={int(float(rr)*100)}%" if rr is not None else "rr=?"
    notice_str = f"notice={int(notice)}d" if notice is not None else "notice=?"
    otw_str    = "open=yes" if otw is True else "open=no"
    score_str  = f"scores[sk={sk:.2f} tr={tr:.2f} av={avail:.2f} en={engage:.2f}]"

    # Header
    header = f"{title} | {yoe_str} | {loc}"

    # Strengths
    strengths: list[str] = []
    if matched:
        strengths.append(f"matches: {', '.join(matched[:4])}")
    if github not in (None, -1) and float(github) > 65:
        strengths.append(gh_str)
    if rr is not None and float(rr) > 0.70:
        strengths.append(rr_str)
    if notice is not None and int(notice) <= 30:
        strengths.append(notice_str)
    if otw is True:
        strengths.append("open-to-work")
    if companies and not any(cf in companies[0] for cf in ["tcs","infosys","wipro","accenture"]):
        strengths.append(f"product-co({companies[0][:20]})")

    # Concerns
    concerns: list[str] = []
    if penalties:
        concerns.append(penalties[0])
    if flags:
        concerns.append(flags[0])
    if missing:
        concerns.append(f"gap: {missing[0]}")
    if rr is not None and float(rr) < 0.30:
        concerns.append(f"low-rr({rr_str})")

    # Build output
    parts = [header]
    if strengths:
        parts.append(f"STRENGTHS: {'; '.join(strengths[:3])}")
    parts.append(score_str)
    parts.append(f"SIGNALS: {gh_str} {rr_str} {notice_str} {otw_str}")
    if concerns:
        parts.append(f"CONCERN: {'; '.join(concerns[:2])}")

    return ". ".join(parts) + "."
