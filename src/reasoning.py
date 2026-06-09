
"""
reasoning.py — DeepShortlist (fixed)
Uses universal field_extractor — no more Unknown role outputs.
"""
from __future__ import annotations
from typing import Any
from src.field_extractor import (
    get_title, get_yoe_str, get_location,
    get_signals, get_company_names,
)

def generate_reasoning(c, scores, rank):
    title     = get_title(c)
    yoe_str   = get_yoe_str(c)
    loc_str   = get_location(c)
    signals   = get_signals(c)
    matched   = scores.get("matched_skills") or []
    missing   = scores.get("missing_skills") or []
    flags     = scores.get("trajectory_flags") or []
    penalties = scores.get("penalty_reasons") or []
    avail     = scores.get("availability_score", 0)
    github    = signals.get("github_activity_score")
    rr_raw    = signals.get("recruiter_response_rate")
    notice    = signals.get("notice_period_days")
    otw       = signals.get("open_to_work_flag", False)
    companies = get_company_names(c)

    rr_str     = f"{int(float(rr_raw)*100)}%" if rr_raw is not None else None
    notice_str = f"{int(notice)}-day notice" if notice is not None else None
    gh_str     = f"{int(github)}/100" if github not in (None, -1) else None

    if scores.get("final_score", 1.0) == 0.0:
        return "Profile excluded: synthetic or internally inconsistent data detected."

    if rank <= 10:
        strengths = []
        if matched: strengths.append(f"strong match on {', '.join(matched[:3])}")
        if gh_str and int(github) > 60: strengths.append(f"active GitHub ({gh_str})")
        if rr_str and float(rr_raw) > 0.70: strengths.append(f"responsive to recruiters ({rr_str})")
        if otw: strengths.append("actively open to work")
        if notice is not None and int(notice) <= 30: strengths.append(f"available quickly ({notice_str})")
        if companies: strengths.append(f"product-company background ({companies[0]})")
        concern = ""
        if missing: concern = f" Gap to probe: {missing[0]}."
        elif not otw: concern = " Verify current availability."
        base = f"{title} | {yoe_str} | {loc_str}."
        if strengths:
            base += f" {strengths[0].capitalize()}"
            if len(strengths) > 1: base += f"; {strengths[1]}"
            base += "."
        return (base + concern).strip()

    if rank <= 30:
        lead = f"{title} ({yoe_str}) shows good alignment"
        if matched: lead += f" on {', '.join(matched[:2])}"
        lead += "."
        concerns = []
        if flags: concerns.append(flags[0])
        if missing: concerns.append(f"missing {missing[0]}")
        if rr_str and float(rr_raw) < 0.40: concerns.append(f"low response rate ({rr_str})")
        c_str = f" Concern: {'; '.join(concerns[:2])}." if concerns else ""
        return (lead + c_str).strip()

    if rank <= 60:
        match_note = f"partial match: {', '.join(matched[:2])}" if matched else "limited skill alignment"
        concerns = []
        if missing: concerns.append(f"lacks {', '.join(missing[:2])}")
        if flags: concerns.append(flags[0])
        if penalties: concerns.append(penalties[0])
        if avail < 0.35: concerns.append("low availability signals")
        return f"{title} ({yoe_str}), {match_note}. Concerns: {'; '.join(concerns[:2])}."

    skill_note = f"matches {len(matched)} required skill(s)" if matched else "minimal skill overlap"
    issues = []
    if penalties: issues.append(penalties[0])
    if flags: issues.append(flags[0])
    if missing: issues.append(f"missing: {', '.join(missing[:2])}")
    issues_str = "; ".join(issues[:2]) if issues else "below threshold"
    return f"{title} ({yoe_str}): {skill_note}. Rank {rank} — {issues_str}; borderline fit."
