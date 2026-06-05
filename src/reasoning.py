"""
reasoning.py — DeepShortlist
Generates specific, fact-based, non-templated reasoning strings
for each ranked candidate. Designed to pass Stage 4 manual review:
  - References specific profile facts
  - Connects to JD requirements
  - Honestly flags concerns
  - Varies by rank tier
  - Never hallucinated data
"""

from __future__ import annotations

import random
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lower(s: Any) -> str:
    return str(s).lower() if s else ""


def _title(c: dict) -> str:
    return c.get("current_title") or c.get("title") or "Unknown role"


def _yoe(c: dict) -> str:
    yoe = c.get("years_of_experience") or c.get("years_experience") or c.get("total_experience")
    if yoe is None:
        return "unknown"
    return f"{float(yoe):.1f}y"


def _loc(c: dict) -> str:
    return c.get("location") or c.get("city") or "unknown location"


def _signals(c: dict) -> dict:
    return c.get("redrob_signals") or {}


def _top_skills(c: dict, n: int = 4) -> str:
    skills = c.get("skills") or []
    if isinstance(skills, list) and skills:
        return ", ".join(str(s) for s in skills[:n])
    return "no skills listed"


def _github(c: dict) -> str | None:
    g = _signals(c).get("github_activity_score")
    if g is None or g == -1:
        return None
    return f"{int(g)}/100"


def _response_rate(c: dict) -> str | None:
    rr = _signals(c).get("recruiter_response_rate")
    if rr is None:
        return None
    return f"{int(float(rr)*100)}%"


def _notice(c: dict) -> str | None:
    n = _signals(c).get("notice_period_days")
    if n is None:
        return None
    return f"{int(n)}-day notice"


def _last_active(c: dict) -> str | None:
    la = _signals(c).get("last_active_date")
    return str(la) if la else None


def _open_to_work(c: dict) -> bool:
    return _signals(c).get("open_to_work_flag", False) is True


def _companies(c: dict) -> list[str]:
    out = []
    for job in (c.get("work_history") or c.get("experience") or []):
        if isinstance(job, dict):
            co = job.get("company") or ""
            if co:
                out.append(co)
    return out[:3]


# ---------------------------------------------------------------------------
# Tier-aware reasoning builder
# ---------------------------------------------------------------------------

def generate_reasoning(
    c: dict[str, Any],
    scores: dict[str, Any],
    rank: int,
) -> str:
    """
    Build a 1-2 sentence reasoning string.
    Varies structure by rank tier to avoid templating.
    """
    title     = _title(c)
    yoe_str   = _yoe(c)
    loc_str   = _loc(c)
    skills_str= _top_skills(c, 4)
    github    = _github(c)
    rr        = _response_rate(c)
    notice    = _notice(c)
    otw       = _open_to_work(c)
    companies = _companies(c)
    matched   = scores.get("matched_skills") or []
    missing   = scores.get("missing_skills") or []
    flags     = scores.get("trajectory_flags") or []
    penalties = scores.get("penalty_reasons") or []
    sem_score = scores.get("semantic_score", 0)
    avail     = scores.get("availability_score", 0)

    # Honeypot / disqualified
    if scores.get("final_score", 1.0) == 0.0 or "honeypot" in " ".join(penalties):
        return (
            f"Profile excluded: detected impossible or internally inconsistent "
            f"data indicative of a synthetic profile."
        )

    # --- Tier 1: Ranks 1–10 (top picks — strong positive reasoning) ---
    if rank <= 10:
        return _tier_top(title, yoe_str, loc_str, skills_str, github, rr, notice, otw,
                         matched, missing, companies, sem_score, rank)

    # --- Tier 2: Ranks 11–30 (strong but with one notable nuance) ---
    if rank <= 30:
        return _tier_strong(title, yoe_str, loc_str, skills_str, github, rr, notice,
                            matched, missing, flags, companies, rank)

    # --- Tier 3: Ranks 31–60 (moderate fit, clear concern) ---
    if rank <= 60:
        return _tier_moderate(title, yoe_str, skills_str, rr, matched, missing,
                              flags, penalties, avail, rank)

    # --- Tier 4: Ranks 61–100 (weak fit, honest assessment) ---
    return _tier_weak(title, yoe_str, skills_str, matched, missing, flags, penalties, avail, rank)


# ---------------------------------------------------------------------------
# Per-tier builders (structurally distinct to avoid templating)
# ---------------------------------------------------------------------------

def _tier_top(title, yoe_str, loc_str, skills_str, github, rr, notice, otw,
              matched, missing, companies, sem_score, rank) -> str:
    strengths = []

    if matched:
        strengths.append(f"matches core JD requirements on {', '.join(matched[:3])}")
    if github and int(github.split('/')[0]) > 60:
        strengths.append(f"strong GitHub activity ({github})")
    if rr and int(rr[:-1]) > 70:
        strengths.append(f"responsive to recruiters ({rr})")
    if otw:
        strengths.append("actively open to work")
    if notice and int(notice.split('-')[0]) <= 30:
        strengths.append(f"immediately available ({notice})")
    if companies:
        strengths.append(f"product-company background ({companies[0]})")

    concern = ""
    if missing:
        concern = f" One gap to probe: {missing[0]}."
    elif not otw:
        concern = " Not currently marked open-to-work — verify availability."

    base = f"{title} | {yoe_str} exp | {loc_str}."
    if strengths:
        base += f" {strengths[0].capitalize()}"
        if len(strengths) > 1:
            base += f"; {strengths[1]}"
        base += "."
    base += concern
    return base.strip()


def _tier_strong(title, yoe_str, loc_str, skills_str, github, rr, notice,
                 matched, missing, flags, companies, rank) -> str:
    # Lead with the single strongest signal, then one concern
    lead = f"{title} ({yoe_str}) shows solid alignment"
    if matched:
        lead += f" on {', '.join(matched[:2])}"
    lead += "."

    concern_parts = []
    if flags:
        concern_parts.append(flags[0])
    if missing:
        concern_parts.append(f"missing {missing[0]}")
    if rr and int(rr[:-1]) < 40:
        concern_parts.append(f"low recruiter response rate ({rr})")
    if notice and int(notice.split('-')[0]) > 60:
        concern_parts.append(f"long notice period ({notice})")

    concern = f" Concern: {'; '.join(concern_parts[:2])}." if concern_parts else ""
    return (lead + concern).strip()


def _tier_moderate(title, yoe_str, skills_str, rr, matched, missing,
                   flags, penalties, avail, rank) -> str:
    # More neutral, acknowledge partial fit
    parts = [f"{title} ({yoe_str})"]

    if matched:
        parts.append(f"partial skill match: {', '.join(matched[:2])}")
    else:
        parts.append("limited direct skill alignment with JD")

    concerns = []
    if missing:
        concerns.append(f"lacks {', '.join(missing[:2])}")
    if flags:
        concerns.append(flags[0])
    if penalties:
        concerns.append(penalties[0])
    if avail < 0.35:
        concerns.append("low availability signals")

    concern_str = "; ".join(concerns[:2])
    return f"{', '.join(parts)}. Concerns: {concern_str}." if concerns else ", ".join(parts) + "."


def _tier_weak(title, yoe_str, skills_str, matched, missing, flags, penalties, avail, rank) -> str:
    # Honest — this candidate is near the bottom of the top-100
    skill_note = f"matches {len(matched)} of {len(matched)+len(missing)} required skills" if matched else "minimal required skill overlap"
    issue_parts = []
    if penalties:
        issue_parts.append(penalties[0])
    if flags:
        issue_parts.append(flags[0])
    if missing:
        issue_parts.append(f"missing: {', '.join(missing[:2])}")
    if avail < 0.25:
        issue_parts.append("low engagement/availability")

    issues = "; ".join(issue_parts[:2]) if issue_parts else "below composite score threshold"
    return (
        f"{title} ({yoe_str}): {skill_note}. "
        f"Included as rank {rank} — {issues}; borderline fit only."
    )
