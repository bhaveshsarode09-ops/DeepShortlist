
"""
Universal field extractor — handles any candidate dataset schema.
Tries 20+ field name variants per field.
"""
from __future__ import annotations
from typing import Any

_TITLE_KEYS = [
    "current_title","title","job_title","designation",
    "current_designation","position","role","role_title",
    "current_role","current_position","profile_title",
    "headline","professional_title","current_job_title",
]
_EXP_KEYS = [
    "years_of_experience","years_experience","total_experience",
    "exp_years","experience_years","work_experience",
    "years_exp","total_exp","yoe","experience",
    "total_work_experience","total_years","career_years",
]
_LOCATION_KEYS = [
    "location","city","current_city","current_location",
    "base_location","home_city","preferred_location",
    "state","region","address","geo","place",
    "residence","hometown","work_location",
]
_SKILLS_KEYS = [
    "skills","skill_set","technical_skills","key_skills",
    "expertise","technologies","tech_stack","competencies",
    "tools","languages","frameworks","skill_list",
    "primary_skills","secondary_skills",
]
_WORK_HISTORY_KEYS = [
    "work_history","experience","work_experience",
    "employment_history","job_history","employment",
    "career_history","positions","jobs","roles",
    "previous_roles","work","employment_records",
]
_EDUCATION_KEYS = [
    "education","academics","qualifications",
    "educational_background","edu","degrees",
]
_BIO_KEYS = [
    "bio","summary","about","about_me","profile_summary",
    "overview","description","professional_summary",
    "career_summary","introduction",
]

def _get(d, keys, default=None):
    for k in keys:
        v = d.get(k)
        if v is not None and v != "" and v != [] and v != {}:
            return v
    return default

def get_title(c):
    return str(_get(c, _TITLE_KEYS) or "").strip() or "Unknown"

def get_yoe(c):
    raw = _get(c, _EXP_KEYS)
    if raw is None: return None
    try: return float(str(raw).split()[0])
    except: return None

def get_yoe_str(c):
    yoe = get_yoe(c)
    return f"{yoe:.1f}y" if yoe is not None else "unknown"

def get_location(c):
    loc = _get(c, _LOCATION_KEYS)
    if loc and isinstance(loc, dict):
        return (loc.get("city") or loc.get("name") or loc.get("state") or "").strip()
    return str(loc or "").strip() or "Unknown"

def get_skills(c):
    raw = _get(c, _SKILLS_KEYS, [])
    if isinstance(raw, list): return [str(s).lower().strip() for s in raw if s]
    if isinstance(raw, str): return [s.strip().lower() for s in raw.split(",") if s.strip()]
    return []

def get_work_history(c):
    raw = _get(c, _WORK_HISTORY_KEYS, [])
    return [j for j in raw if isinstance(j, dict)] if isinstance(raw, list) else []

def get_education(c):
    raw = _get(c, _EDUCATION_KEYS, [])
    return [e for e in raw if isinstance(e, dict)] if isinstance(raw, list) else []

def get_bio(c):
    return str(_get(c, _BIO_KEYS) or "").strip()[:500]

def get_signals(c):
    return c.get("redrob_signals") or c.get("signals") or c.get("platform_signals") or {}

def get_company_names(c):
    out = []
    for job in get_work_history(c):
        co = (job.get("company") or job.get("employer") or
              job.get("organization") or job.get("organisation") or "")
        if co: out.append(str(co).lower().strip())
    return out[:5]

def get_job_titles(c):
    titles = [get_title(c).lower()]
    for job in get_work_history(c):
        t = (job.get("title") or job.get("role") or
             job.get("designation") or job.get("position") or "")
        if t: titles.append(str(t).lower().strip())
    return titles

def build_rich_text(c):
    parts = []
    title = get_title(c)
    if title and title != "Unknown": parts.extend([title, title])
    yoe = get_yoe(c)
    if yoe is not None: parts.append(f"{yoe:.0f} years experience")
    skills = get_skills(c)
    if skills:
        s = " ".join(skills)
        parts.extend([s, s])
    for job in get_work_history(c)[:5]:
        t = job.get("title") or job.get("role") or job.get("designation") or ""
        co = job.get("company") or job.get("employer") or job.get("organization") or ""
        desc = (job.get("description") or job.get("summary") or
                job.get("responsibilities") or "")
        if t: parts.append(str(t))
        if co: parts.append(str(co))
        if desc: parts.append(str(desc)[:300])
    for edu in get_education(c)[:2]:
        deg = edu.get("degree") or edu.get("qualification") or ""
        field = edu.get("field") or edu.get("specialization") or ""
        inst = edu.get("institution") or edu.get("university") or ""
        parts.append(f"{deg} {field} {inst}".strip())
    bio = get_bio(c)
    if bio: parts.append(bio)
    loc = get_location(c)
    if loc and loc != "Unknown": parts.append(loc)
    return " ".join(p for p in parts if p and str(p).strip())
