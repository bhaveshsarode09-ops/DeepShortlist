# DeepShortlist v2

**Intelligent Candidate Ranking — India Runs Hackathon 2026 · Redrob AI · Track 1**

---

## Architecture

```
candidates.jsonl.gz (100K)
        │
        ▼
┌──────────────────────────────────┐
│  Stage 1: Dynamic Schema         │  Auto-detects field names at runtime
│  Detection                       │  Works with ANY dataset schema
│  (one-time, <1s)                 │  No hardcoded field names
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│  Stage 2: Field-Weighted BM25+   │  100K → top 3000 in ~5s
│  First Pass                      │  Title×3, Skills×2.5, Desc×1
│  (Lv & Zhai, 2011)               │  BM25+ lower-bound δ=0.5
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│  Stage 3: LSA Reranking          │  TF-IDF + TruncatedSVD (120-dim)
│  (Deerwester et al., 1990)       │  Captures latent semantic structure
│  bigram TF-IDF, 40K vocab        │  Cosine similarity in latent space
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│  Stage 4: RRF Ensemble           │  Reciprocal Rank Fusion (k=60)
│  (Cormack et al., 2009)          │  4 signals: retrieval·skill·avail·traj
│  + Percentile Spread             │  Sigmoid stretch → wide score range
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│  Stage 5: Structured Output      │  AI-parseable reasoning
│  Reasoning + CSV                 │  NDCG/MRR optimised ranking
└────────────┬─────────────────────┘
             │
             ▼
      team_xxx.csv (top-100)
```

---

## Algorithms

| Component | Algorithm | Why |
|---|---|---|
| First pass | **BM25+** (Lv & Zhai 2011) | Handles term saturation; δ lower bound prevents zero scores |
| Reranking | **LSA / TruncatedSVD** | Captures synonyms: "dense retrieval" ≈ "embedding-based search" |
| Fusion | **RRF** (Cormack 2009) | More robust than weighted sum; handles score scale differences |
| Distribution | **Percentile sigmoid stretch** | Guarantees wide score range regardless of input clustering |
| Penalty | **Pre-RRF multipliers** | Applied before ranking so wrong-domain candidates truly sink |
| Honeypot | **5-rule anomaly detection** | Statistical + rule-based impossible profile detection |
| Schema | **Dynamic key scoring** | Scores every key against expected patterns; no hardcoded names |

---

## Scoring Signals

| Signal | Weight (RRF) | What it captures |
|---|---|---|
| BM25+/LSA Retrieval | 35% | Semantic JD-profile relevance |
| Skill Coverage | 28% | Required + preferred skill matching |
| Availability | 22% | 23 behavioral signals: open-to-work, recency, notice, response rate |
| Career Trajectory | 15% | Domain alignment, experience bracket, product-company background |

### Penalty Multipliers (applied before RRF)
| Condition | Multiplier |
|---|---|
| Wrong job family (HR, Content Writer, etc.) | 0.35× |
| Consulting-only background | 0.55× |
| Wrong domain (CV/speech, no NLP/IR) | 0.60× |
| Insufficient experience | 0.70× |
| Honeypot detected | 0.00× |

---

## Performance

| Metric | Value |
|---|---|
| Speed (100K candidates) | ~13 seconds |
| RAM usage | ~200 MB peak |
| Model downloads | **Zero** |
| Internet required | **No** |
| GPU required | **No** |
| Dependencies | scikit-learn, scipy, pandas, numpy |

---

## Setup

```bash
pip install scikit-learn scipy pandas numpy
```

## Run

```bash
python rank.py --candidates candidates.jsonl.gz --out team_xxx.csv
```

## Validate

```bash
python validate_submission.py team_xxx.csv
```

---

## Project Structure

```
DeepShortlist/
├── rank.py                    # Single command entry point
├── validate_submission.py     # Official format validator
├── requirements.txt
├── README.md
├── submission_metadata.yaml
└── src/
    ├── schema.py              # Dynamic schema detection
    ├── retrieval.py           # BM25+ + LSA engine
    ├── features.py            # Skill/trajectory/availability/engagement
    ├── scorer.py              # RRF ensemble + percentile spread
    ├── reasoning.py           # AI-parseable reasoning
    └── ranker.py              # Pipeline orchestration
```

---

## Author

**Bhavesh Sarode** · India Runs Hackathon 2026 · Solo
