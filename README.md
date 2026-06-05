#  DeepShortlist

**Intelligent Candidate Ranking System**  
*India Runs Hackathon 2026 · Redrob AI · Track 1: Data & AI Challenge*

---

## Overview

DeepShortlist ranks candidates from a 100K pool against a Senior AI Engineer JD using a **4-stage CPU-native pipeline** that goes far beyond keyword matching — combining semantic embeddings, 23 behavioral signals, career trajectory analysis, and multi-rule honeypot detection.

**Core insight:** The JD explicitly warns that keyword stuffers will rank high on naive systems. DeepShortlist is built to beat that trap.

---

## Architecture

```
candidates.jsonl.gz (100K)
        │
        ▼
┌─────────────────────────────┐
│  Stage 1: FAISS First Pass  │  sentence-transformers → cosine similarity
│  all-MiniLM-L6-v2 (384-dim) │  → top-2000 by semantic score
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Stage 2: Multi-Signal Full │  6 scoring dimensions:
│  Scoring (top-2000 pool)    │  Semantic(30%) + Skill(20%) +
│                             │  Trajectory(15%) + Availability(20%) +
│                             │  Engagement(10%) + Location(5%)
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Stage 3: Honeypot Filter   │  6 detection rules:
│  + Penalty Multipliers      │  experience vs company age,
│                             │  signal envelope violations,
│                             │  temporal impossibility, etc.
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Stage 4: Reasoning         │  Fact-based, tier-aware,
│  Generation                 │  non-templated reasoning per candidate
└────────────┬────────────────┘
             │
             ▼
      team_xxx.csv (top-100)
```

---

## Scoring Signals

### 6 Scoring Dimensions

| Dimension | Weight | What it captures |
|---|---|---|
| **Semantic Similarity** | 30% | JD-profile cosine similarity via transformer embeddings |
| **Skill Gap** | 20% | Required + preferred skill coverage vs JD |
| **Career Trajectory** | 15% | Domain alignment, seniority, product-company background |
| **Availability** | 20% | 8 of 23 behavioral signals: open-to-work, recency, notice period, response rate |
| **Engagement** | 10% | GitHub activity, saved by recruiters, skill assessments, endorsements |
| **Location** | 5% | Pune/Noida/Hyderabad/Mumbai/Delhi NCR or willing to relocate |

### Penalty Multipliers

| Condition | Multiplier |
|---|---|
| Wrong job family (HR Manager, Graphic Designer, etc.) | 0.40× |
| Pure consulting background (TCS, Infosys, etc.) | 0.55× |
| Research-only profile without production signals | 0.60× |
| CV/speech/robotics primary without NLP/IR | 0.65× |
| Experience below hard minimum | 0.75× |
| Honeypot detected | 0.00× |

### Honeypot Detection (6 Rules)

1. **Company lifespan** — tenure > company founding age (e.g. 8y at a 3y-old startup)
2. **Skill explosion** — 40+ skills listed or all assessments ≥ 98/100
3. **Signal envelope violations** — values outside declared ranges
4. **Temporal impossibility** — signup year before graduation year
5. **Too-perfect profile** — 99th percentile across 5+ independent dimensions simultaneously
6. **Experience bounds** — `years_of_experience` > 30 or < 0

---

## Compute Constraints Compliance

| Constraint | Limit | DeepShortlist |
|---|---|---|
| Runtime | ≤ 5 min | ~2–3 min on 16 GB CPU |
| Memory | ≤ 16 GB | ~2.5 GB peak (embeddings matrix) |
| Compute | CPU only |  No GPU used |
| Network | Off during ranking |  All models local |

**Speed breakdown:**
- FAISS index build: ~90s for 100K candidates (batch encoding)
- Full scoring of top-2000: ~5s (vectorised Python)
- Total pipeline: ~120s on a modern CPU

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt

# CPU-only PyTorch (smaller install):
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. Download the embedding model (pre-computation, one-time)

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

---

## Run

### Full pipeline (single command — as required by submission_spec.md)

```bash
python rank.py --candidates ./candidates.jsonl.gz --out ./team_xxx.csv
```

### With a gzipped file

```bash
python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv
```

### With plain JSONL

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

### Validate your submission

```bash
python validate_submission.py team_xxx.csv
```

---

## Project Structure

```
DeepShortlist/
├── rank.py                     # Single-command entry point
├── validate_submission.py      # Official format validator (from bundle)
├── requirements.txt
├── README.md
├── submission_metadata.yaml
└── src/
    ├── __init__.py
    ├── jd_config.py            # JD constants: skills, weights, disqualifiers
    ├── embeddings.py           # FAISS + sentence-transformers engine
    ├── signals.py              # 23 behavioral signal processor
    ├── scorer.py               # Multi-signal weighted scorer + penalties
    ├── honeypot.py             # 6-rule honeypot detector
    ├── reasoning.py            # Fact-based reasoning generator
    └── ranker.py               # Pipeline orchestration
```

---

## Key Design Decisions

**Why FAISS first-pass?**  
Encoding 100K candidates with sentence-transformers takes ~90s. Running full multi-signal scoring on all 100K would take too long. FAISS retrieves top-2000 by semantic similarity in milliseconds, letting us apply the expensive scoring only where it matters.

**Why 20% weight on availability?**  
The JD explicitly states: *"A perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% response rate is, for hiring purposes, not actually available."* Most teams will underweight this. We don't.

**Why 30% weight on semantic (not higher)?**  
Pure semantic similarity would rank keyword stuffers and candidates who mention AI terms in passing. Capping at 30% forces the model to look at real career trajectory and behavioral signals.

**Why honeypot suspicion threshold at 0.60?**  
Conservative enough to avoid false positives, strict enough to catch all 6 detection patterns. A candidate flagged by 2+ rules reliably crosses 0.60.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector Search | `faiss-cpu` (IndexFlatIP) |
| Data | `pandas`, `numpy` |
| Scoring | Pure Python + NumPy (vectorised) |
| No external APIs | Fully offline during ranking |

---

## Author

**Bhavesh Sarode**  
India Runs Hackathon 2026 · Solo Participant  
GitHub: [bhaveshsarode09-ops](https://github.com/bhaveshsarode09-ops)
