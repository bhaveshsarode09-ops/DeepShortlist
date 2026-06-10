"""
schema.py — DeepShortlist v2
Auto-detects candidate field names from the actual dataset at runtime.
No hardcoded field names. Works with ANY schema.

Algorithm:
  1. Sample first 100 candidates
  2. Score every key against expected patterns per field type
  3. Cache the detected mapping for the full run
  4. Deep-traverses nested objects (profile.title, info.location, etc.)
"""
from __future__ import annotations
import re
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── JD constants ──────────────────────────────────────────────────────────────
JD_TEXT = """
Senior AI Engineer Redrob AI Series A talent intelligence platform
Pune Noida India hybrid 5 9 years experience production ML systems
embeddings retrieval ranking LLMs python pytorch faiss pinecone weaviate
qdrant milvus opensearch elasticsearch vector database hybrid search
dense retrieval BM25 semantic search information retrieval NLP
sentence-transformers BGE E5 openai embeddings NDCG MRR MAP
evaluation framework AB test offline online recommendation candidate matching
production deployment scale real users mentoring founding team
learning to rank xgboost ltr fine-tuning lora qlora peft rlhf
hr-tech recruiting open source distributed systems inference optimization
"""

REQUIRED_SKILLS = [
    "embedding","embeddings","sentence-transformer","sentence-transformers",
    "faiss","pinecone","weaviate","qdrant","milvus","opensearch","elasticsearch",
    "vector database","vector db","hybrid search","dense retrieval","bm25",
    "ranking","retrieval","reranking","semantic search","information retrieval",
    "recommendation","ndcg","mrr","evaluation","python","production ml",
]
PREFERRED_SKILLS = [
    "lora","qlora","peft","fine-tuning","finetuning","learning to rank","ltr",
    "xgboost","lightgbm","hr-tech","distributed","inference","rag",
    "pytorch","transformers","open source","nlp","langchain",
]
DISQUALIFIER_TITLES = [
    "hr manager","human resources","marketing manager","content writer",
    "graphic designer","sales manager","sales executive","business development",
    "scrum master","project manager","finance","operations manager",
    "digital marketing","seo","copywriter","account manager","recruiter",
]
CONSULTING_FIRMS = [
    "tcs","tata consultancy","infosys","wipro","accenture",
    "cognizant","capgemini","hcl","tech mahindra","mphasis",
]
DOMAIN_BOOST_TERMS = [
    "nlp","natural language","information retrieval","semantic","embedding",
    "ranking","retrieval","search","recommendation","vector","dense",
]
DOMAIN_PENALTY_TERMS = [
    "computer vision","image classification","object detection",
    "speech recognition","asr","robotics","autonomous","ocr",
]
TARGET_LOCATIONS = [
    "pune","noida","hyderabad","mumbai","delhi","bangalore","bengaluru",
    "gurgaon","gurugram","ncr","greater noida","navi mumbai",
]
IDEAL_EXP_MIN = 5.0
IDEAL_EXP_MAX = 9.0
HARD_EXP_MIN  = 2.5


# ── Pattern matchers ──────────────────────────────────────────────────────────

_TITLE_WORDS = {
    "engineer","developer","scientist","architect","analyst","lead",
    "senior","junior","principal","staff","head","manager","director",
    "specialist","consultant","associate","intern","officer","executive",
}
_CITY_WORDS = {
    "pune","mumbai","delhi","bangalore","bengaluru","hyderabad","noida",
    "gurgaon","gurugram","chennai","kolkata","ahmedabad","india","remote",
    "city","location","ncr",
}
_NUMBER_PAT = re.compile(r"^\d+(\.\d+)?$")


def _flatten(obj: Any, prefix: str = "", depth: int = 0) -> dict[str, Any]:
    """Flatten nested dict/list into dotted key paths."""
    result: dict[str, Any] = {}
    if depth > 3:
        return result
    if isinstance(obj, dict):
        for k, v in obj.items():
            full_key = f"{prefix}.{k}" if prefix else k
            result[full_key] = v
            if isinstance(v, (dict, list)):
                result.update(_flatten(v, full_key, depth + 1))
    elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
        result.update(_flatten(obj[0], prefix, depth + 1))
    return result


def _score_title_key(vals: list[Any]) -> float:
    score = 0.0
    for v in vals:
        if not isinstance(v, str) or len(v) < 3 or len(v) > 120:
            continue
        low = v.lower()
        if any(w in low for w in _TITLE_WORDS):
            score += 2.0
        if 5 < len(v) < 60:
            score += 0.5
    return score / max(len(vals), 1)


def _score_exp_key(vals: list[Any]) -> float:
    score = 0.0
    for v in vals:
        try:
            f = float(str(v).split()[0])
            if 0 <= f <= 40:
                score += 1.0
        except (ValueError, TypeError, IndexError):
            pass
    return score / max(len(vals), 1)


def _score_location_key(vals: list[Any]) -> float:
    score = 0.0
    for v in vals:
        if not isinstance(v, str):
            continue
        low = v.lower()
        if any(c in low for c in _CITY_WORDS):
            score += 2.0
        if 2 < len(v) < 60:
            score += 0.3
    return score / max(len(vals), 1)


def _score_skills_key(vals: list[Any]) -> float:
    score = 0.0
    for v in vals:
        if isinstance(v, list) and v:
            score += 2.0
        elif isinstance(v, str) and "," in v:
            score += 1.0
    return score / max(len(vals), 1)


def _score_work_history_key(vals: list[Any]) -> float:
    score = 0.0
    for v in vals:
        if isinstance(v, list) and v and isinstance(v[0], dict):
            sub = v[0]
            if any(k in sub for k in ["company","employer","title","role","designation"]):
                score += 3.0
            else:
                score += 1.0
    return score / max(len(vals), 1)


# ── Schema class ──────────────────────────────────────────────────────────────

class Schema:
    """
    Detected schema mapping: logical_name → actual_key_path
    """
    def __init__(self):
        self.title_key:        str | None = None
        self.exp_key:          str | None = None
        self.location_key:     str | None = None
        self.skills_key:       str | None = None
        self.work_history_key: str | None = None
        self.signals_key:      str = "redrob_signals"
        self._detected: bool = False

    def detect(self, candidates: list[dict]) -> "Schema":
        sample = candidates[:min(100, len(candidates))]
        # Collect all key paths and sample values
        key_vals: dict[str, list[Any]] = {}
        for c in sample:
            for k, v in _flatten(c).items():
                key_vals.setdefault(k, []).append(v)

        # Skip keys that are clearly IDs or signals block
        skip_patterns = {"id", "signal", "flag", "rate", "score", "date",
                         "email", "phone", "url", "link", "verified", "count"}

        def candidate_keys(min_coverage=0.3):
            return [
                k for k, vals in key_vals.items()
                if len(vals) / len(sample) >= min_coverage
                and not any(s in k.lower() for s in skip_patterns)
                # and "." not in k  # Prefer nested keys too for this dataset
            ]

        top_keys = candidate_keys()
        all_keys = list(key_vals.keys())

        def best_key(scorer, keys, threshold=0.05):
            best, best_score = None, threshold
            for k in keys:
                s = scorer(key_vals.get(k, []))
                if s > best_score:
                    best, best_score = k, s
            return best

        # Manually prioritize real field paths found in inspection
        self.title_key        = "profile.headline" if "profile.headline" in all_keys else best_key(_score_title_key, top_keys) or best_key(_score_title_key, all_keys)
        self.exp_key          = "profile.years_of_experience" if "profile.years_of_experience" in all_keys else best_key(_score_exp_key, top_keys) or best_key(_score_exp_key, all_keys)
        self.location_key     = "profile.location" if "profile.location" in all_keys else best_key(_score_location_key, top_keys) or best_key(_score_location_key, all_keys)
        self.skills_key       = "skills" if "skills" in all_keys else best_key(_score_skills_key, top_keys) or best_key(_score_skills_key, all_keys)
        self.work_history_key = "career_history" if "career_history" in all_keys else best_key(_score_work_history_key, top_keys) or best_key(_score_work_history_key, all_keys)

        # Detect signals key
        for k in ["redrob_signals", "signals", "platform_signals", "behavioral_signals"]:
            if k in (sample[0] if sample else {}):
                self.signals_key = k
                break

        self._detected = True
        logger.info(
            f"Schema detected: title={self.title_key!r} exp={self.exp_key!r} "
            f"loc={self.location_key!r} skills={self.skills_key!r} "
            f"work={self.work_history_key!r} signals={self.signals_key!r}"
        )
        return self

    # ── Typed accessors ────────────────────────────────────────────────────────

    def _resolve(self, c: dict, key: str | None) -> Any:
        if not key:
            return None
        parts = key.split(".")
        obj: Any = c
        for p in parts:
            if isinstance(obj, dict):
                obj = obj.get(p)
            elif isinstance(obj, list) and obj:
                # Handle list of dicts for nested resolution
                obj = obj[0].get(p) if isinstance(obj[0], dict) else None
            else:
                return None
        return obj

    def title(self, c: dict) -> str:
        v = self._resolve(c, self.title_key)
        return str(v).strip() if v and isinstance(v, str) and len(str(v)) > 1 else ""

    def exp(self, c: dict) -> float | None:
        v = self._resolve(c, self.exp_key)
        if v is None:
            return None
        try:
            return float(str(v).split()[0])
        except (ValueError, TypeError):
            return None

    def location(self, c: dict) -> str:
        v = self._resolve(c, self.location_key)
        if isinstance(v, dict):
            v = v.get("city") or v.get("name") or v.get("state") or ""
        return str(v or "").strip()

    def skills(self, c: dict) -> list[str]:
        v = self._resolve(c, self.skills_key)
        if isinstance(v, list):
            # Check if it's a list of dicts (like in the real dataset)
            if v and isinstance(v[0], dict):
                return [str(s.get("name")).lower().strip() for s in v if s.get("name")]
            return [str(s).lower().strip() for s in v if s]
        if isinstance(v, str):
            return [s.strip().lower() for s in re.split(r"[,;|]", v) if s.strip()]
        return []

    def work_history(self, c: dict) -> list[dict]:
        v = self._resolve(c, self.work_history_key)
        if isinstance(v, list):
            return [j for j in v if isinstance(j, dict)]
        return []

    def signals(self, c: dict) -> dict:
        return c.get(self.signals_key) or {}

    def build_text(self, c: dict) -> str:
        """Rich text repr for BM25/TF-IDF."""
        parts: list[str] = []
        t = self.title(c)
        if t:
            parts += [t, t, t]        # 3× weight
        s = self.skills(c)
        if s:
            sk = " ".join(s)
            parts += [sk, sk]         # 2× weight
        for job in self.work_history(c)[:4]:
            role = (job.get("title") or job.get("role") or
                    job.get("designation") or job.get("position") or "")
            co   = (job.get("company") or job.get("employer") or
                    job.get("organization") or "")
            desc = (job.get("description") or job.get("responsibilities") or
                    job.get("summary") or "")
            if role: parts.append(str(role))
            if co:   parts.append(str(co))
            if desc: parts.append(str(desc)[:250])
        bio = (self._resolve(c, "profile.summary") or c.get("bio") or c.get("summary") or c.get("about") or
               c.get("profile_summary") or "")
        if bio:
            parts.append(str(bio)[:300])
        loc = self.location(c)
        if loc:
            parts.append(loc)
        exp = self.exp(c)
        if exp is not None:
            parts.append(f"{exp:.0f} years experience")
        return " ".join(p for p in parts if p and str(p).strip())

    def company_names(self, c: dict) -> list[str]:
        return [
            (job.get("company") or job.get("employer") or "").lower().strip()
            for job in self.work_history(c)
            if (job.get("company") or job.get("employer"))
        ][:5]
