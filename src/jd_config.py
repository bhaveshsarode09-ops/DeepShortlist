"""
jd_config.py — DeepShortlist
All JD-specific constants derived from job_description.md.
Centralised here so changing the JD requires edits in one place only.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Full JD text used for semantic embedding
# ---------------------------------------------------------------------------
JD_TEXT = """
Senior AI Engineer — Founding Team at Redrob AI.
Series A AI-native talent intelligence platform. Pune/Noida India Hybrid.
5-9 years experience. Production ML systems, embeddings, retrieval, ranking, LLMs.

Role: Own the intelligence layer — ranking, retrieval, matching systems.
Build v2 ranking system with embeddings, hybrid retrieval, LLM-based reranking.
Evaluation infrastructure: NDCG, MRR, MAP, A/B testing, recruiter-feedback loops.
Candidate-JD matching at scale. Mentor engineers. Work with PM on product direction.

Scrappy product-engineering attitude. Ship working ranker in a week.
Think about systems not frameworks. Production code writer not architect only.
Async-first, writes a lot, disagrees openly.

Required: embeddings-based retrieval systems sentence-transformers OpenAI embeddings BGE E5.
Vector databases hybrid search Pinecone Weaviate Qdrant Milvus OpenSearch Elasticsearch FAISS.
Strong Python production code quality.
Evaluation frameworks ranking systems NDCG MRR MAP offline-to-online A/B test.

Nice to have: LLM fine-tuning LoRA QLoRA PEFT.
Learning-to-rank XGBoost neural ranking.
HR-tech recruiting marketplace products.
Distributed systems large-scale inference optimization.
Open-source contributions AI ML space.

Ideal: 6-8 years, 4-5 in applied ML at product companies.
Shipped end-to-end ranking search recommendation to real users at scale.
Strong opinions on retrieval hybrid vs dense, evaluation offline vs online, LLM fine-tune vs prompt.
Located in or willing to relocate to Noida or Pune.
Active on Redrob platform, in job market.

Information retrieval NLP semantic search dense retrieval BM25 hybrid search reranking.
Recommendation systems personalization candidate matching talent intelligence.
Python PyTorch TensorFlow scikit-learn pandas numpy.
Production deployment scalability latency quality tradeoffs.
"""

# ---------------------------------------------------------------------------
# Skill taxonomy
# ---------------------------------------------------------------------------

# Must-have skills (each matched skill adds significant score)
REQUIRED_SKILLS: list[str] = [
    # Embeddings / models
    "embedding", "embeddings", "sentence-transformer", "sentence transformers",
    "bge", "e5", "openai embedding",
    # Vector / hybrid search
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "vector database", "vector db", "vector store",
    "hybrid search", "dense retrieval", "sparse retrieval", "bm25",
    # Ranking / retrieval
    "ranking", "retrieval", "reranking", "re-ranking", "semantic search",
    "information retrieval", "recommendation", "candidate matching",
    # Evaluation
    "ndcg", "mrr", "map", "mean average precision", "a/b test",
    "offline evaluation", "evaluation framework",
    # Core language
    "python",
    # Production ML
    "production ml", "production deployment", "mlops", "model serving",
]

# Nice-to-have skills (boost score but not required)
PREFERRED_SKILLS: list[str] = [
    "lora", "qlora", "peft", "fine-tuning", "fine tuning", "finetuning",
    "rlhf", "sft", "instruction tuning",
    "learning to rank", "ltr", "lambdamart", "xgboost", "lightgbm",
    "hr tech", "hr-tech", "recruiting", "talent acquisition", "ats",
    "distributed system", "kafka", "spark", "ray", "dask",
    "inference optimization", "quantization", "onnx", "tensorrt",
    "open source", "github", "hugging face", "transformers",
    "pytorch", "tensorflow", "jax",
    "nlp", "natural language processing", "text classification",
    "langchain", "llamaindex", "rag",
]

# ---------------------------------------------------------------------------
# Hard disqualifiers — applied as multiplier penalties (not zero, but heavy)
# ---------------------------------------------------------------------------

# Job titles that are almost certainly wrong domain
DISQUALIFIER_TITLES: list[str] = [
    "hr manager", "human resources manager", "hr executive",
    "talent acquisition", "recruiter",          # ironically not a fit
    "marketing manager", "digital marketing", "seo", "content writer",
    "content creator", "copywriter", "technical writer",
    "graphic designer", "ux designer", "ui designer", "product designer",
    "sales manager", "sales executive", "business development",
    "account manager", "account executive",
    "scrum master", "project manager", "program manager",
    "finance manager", "finance analyst", "chartered accountant",
    "operations manager", "supply chain",
]

# Consulting-only companies (heavy penalty if entire career is here)
PURE_CONSULTING_COMPANIES: list[str] = [
    "tcs", "tata consultancy services",
    "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl technologies", "hcl tech",
    "tech mahindra", "mphasis", "hexaware", "ltimindtree",
    "persistent systems", "coforge", "niit technologies",
]

# Domains that are disqualifying if they are the *primary* focus
DISQUALIFIER_DOMAINS: list[str] = [
    "computer vision", "image classification", "object detection",
    "image segmentation", "face recognition", "ocr",
    "speech recognition", "asr", "text to speech", "tts",
    "robotics", "autonomous vehicles", "self-driving",
    "signal processing", "time series forecasting",      # adjacent but not core
]

# Research-only indicators
RESEARCH_ONLY_INDICATORS: list[str] = [
    "phd researcher", "research scientist", "principal researcher",
    "research engineer", "postdoc", "postdoctoral",
    "academic researcher", "university research",
]

# ---------------------------------------------------------------------------
# Location config
# ---------------------------------------------------------------------------

TARGET_LOCATIONS: list[str] = [
    "pune", "noida", "hyderabad", "mumbai", "delhi", "delhi ncr",
    "bengaluru", "bangalore", "gurgaon", "gurugram", "ncr",
    "greater noida", "faridabad", "navi mumbai", "thane",
]

# ---------------------------------------------------------------------------
# Experience config
# ---------------------------------------------------------------------------
IDEAL_MIN_EXP = 5.0   # years
IDEAL_MAX_EXP = 9.0
HARD_MIN_EXP  = 3.0   # below this is disqualifying
HARD_MAX_EXP  = 15.0  # above this gets diminishing returns

# ---------------------------------------------------------------------------
# Notice period config (JD prefers <30 days)
# ---------------------------------------------------------------------------
IDEAL_NOTICE_DAYS   = 30
BUYOUT_NOTICE_DAYS  = 30   # company can buy out up to 30 days
PENALTY_NOTICE_DAYS = 90   # 90+ day notice is heavily penalised

# ---------------------------------------------------------------------------
# Scoring weights (must sum to 1.0)
# ---------------------------------------------------------------------------
WEIGHTS: dict[str, float] = {
    "semantic":     0.30,   # JD-profile semantic similarity
    "skill_gap":    0.20,   # Required vs candidate skills
    "trajectory":   0.15,   # Career progression & domain fit
    "availability": 0.20,   # Behavioral: can we actually hire them?
    "engagement":   0.10,   # Platform engagement & GitHub activity
    "location":     0.05,   # Location / relocation fit
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

# ---------------------------------------------------------------------------
# Penalty multipliers (applied AFTER weighted sum)
# ---------------------------------------------------------------------------
PENALTIES: dict[str, float] = {
    "wrong_title":           0.40,   # Clearly wrong job family
    "consulting_only":       0.55,   # Pure services background
    "wrong_domain":          0.65,   # CV/speech/robotics primary
    "research_only":         0.60,   # No production deployment
    "honeypot":              0.00,   # Impossible profile
    "no_production_18mo":    0.70,   # No production code in 18 months
    "experience_too_low":    0.75,   # <3 years
}

# ---------------------------------------------------------------------------
# FAISS / embedding config
# ---------------------------------------------------------------------------
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"   # 384-dim, fast on CPU
EMBEDDING_BATCH   = 512                   # candidates per batch
FAISS_FIRST_PASS  = 2000                  # top-K from FAISS before full scoring
FINAL_TOP_K       = 100                   # final submission size
