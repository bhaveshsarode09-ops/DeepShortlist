#!/usr/bin/env python3
"""
rank.py — DeepShortlist
Single-command entry point required by submission_spec.md Section 10.3.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
    python rank.py --candidates ./candidates.jsonl.gz --out ./team_xxx.csv --top-k 100

Pre-computation (embeddings, if needed separately):
    python rank.py --precompute --candidates ./candidates.jsonl

Compute constraints (per submission_spec.md Section 3):
    - Runtime  ≤ 5 min
    - Memory   ≤ 16 GB RAM
    - CPU only — no GPU
    - No network calls during ranking
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

# Validate that required packages are available before heavy imports
try:
    from sentence_transformers import SentenceTransformer  # noqa: F401
    import faiss                                           # noqa: F401
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

from src.ranker import DeepShortlistRanker

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Candidate loader
# ---------------------------------------------------------------------------

def load_candidates(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        logger.error(f"Candidates file not found: {path}")
        sys.exit(1)

    logger.info(f"Loading candidates from {p.name}  ({p.stat().st_size / 1e6:.1f} MB)…")
    t0 = time.time()
    candidates = []

    if p.suffix == ".gz":
        opener = gzip.open
        mode   = "rt"
    else:
        opener = open
        mode   = "r"

    with opener(str(p), mode, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    candidates.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed line: {e}")

    logger.info(f"Loaded {len(candidates):,} candidates in {time.time()-t0:.1f}s")
    return candidates


# ---------------------------------------------------------------------------
# Output validator (mirrors validate_submission.py logic)
# ---------------------------------------------------------------------------

def quick_validate(df: pd.DataFrame, out_path: str, top_k: int = 100) -> bool:
    """Quick sanity check before saving."""
    ok = True

    if len(df) != top_k:
        logger.error(f"Expected {top_k} rows, got {len(df)}")
        ok = False

    if set(df["rank"].tolist()) != set(range(1, top_k + 1)):
        logger.error(f"Ranks 1-{top_k} not all present exactly once")
        ok = False

    if df["candidate_id"].duplicated().any():
        logger.error("Duplicate candidate_ids detected")
        ok = False

    scores = df["score"].values
    for i in range(len(scores) - 1):
        if scores[i] < scores[i + 1]:
            logger.error(f"Score not non-increasing at ranks {i+1}-{i+2}")
            ok = False
            break

    if not out_path.endswith(".csv"):
        logger.error("Output file must have .csv extension")
        ok = False

    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepShortlist — Intelligent Candidate Ranking System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rank.py --candidates candidates.jsonl.gz --out team_xxx.csv
  python rank.py --candidates sample_candidates.json --out test_out.csv --top-k 100
        """,
    )
    parser.add_argument(
        "--candidates", "-c",
        required=True,
        help="Path to candidates.jsonl or candidates.jsonl.gz",
    )
    parser.add_argument(
        "--out", "-o",
        required=True,
        help="Output CSV path (must end in .csv, filename = your participant ID)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of top candidates to output (default: 100)",
    )
    parser.add_argument(
        "--jd",
        default=None,
        help="Path to custom JD text file (optional, uses embedded JD if not provided)",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model name (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--first-pass",
        type=int,
        default=2000,
        help="FAISS first-pass pool size (default: 2000)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    wall_start = time.time()

    logger.info("=" * 62)
    logger.info("  DeepShortlist — Intelligent Candidate Ranking")
    logger.info("  India Runs Hackathon 2026 | Redrob AI Challenge")
    logger.info("=" * 62)
    logger.info(f"  Candidates : {args.candidates}")
    logger.info(f"  Output     : {args.out}")
    logger.info(f"  Top-K      : {args.top_k}")
    logger.info(f"  Model      : {args.model}")
    logger.info(f"  First pass : {args.first_pass}")
    logger.info("=" * 62)

    # Load JD override if provided
    jd_text = None
    if args.jd:
        jd_path = Path(args.jd)
        if jd_path.exists():
            jd_text = jd_path.read_text(encoding="utf-8")
            logger.info(f"Using JD from file: {args.jd}")
        else:
            logger.warning(f"JD file not found: {args.jd} — using embedded JD")

    # Load candidates
    candidates = load_candidates(args.candidates)

    # Run ranking pipeline
    ranker = DeepShortlistRanker(
        model_name=args.model,
        jd_text=jd_text,
        faiss_first_pass=args.first_pass,
    )
    results_df = ranker.rank(candidates, top_k=args.top_k)

    # Validate before saving
    logger.info("Validating output…")
    if not quick_validate(results_df, args.out, top_k=args.top_k):
        logger.error("Validation failed — output not saved. Check errors above.")
        sys.exit(1)
    logger.info("Validation passed ✓")

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(str(out_path), index=False)

    wall_elapsed = time.time() - wall_start

    logger.info("=" * 62)
    logger.info(f"  Output saved : {out_path}")
    logger.info(f"  Total time   : {wall_elapsed:.1f}s")
    logger.info(f"  Rows         : {len(results_df)}")
    logger.info("=" * 62)
    logger.info("Top 10 candidates:")
    for _, row in results_df.head(10).iterrows():
        logger.info(
            f"  #{int(row['rank']):>3}  {row['candidate_id']}  "
            f"score={row['score']:.4f}  {row['reasoning'][:60]}…"
        )
    logger.info("=" * 62)

    if wall_elapsed > 290:
        logger.warning(
            f"Total runtime {wall_elapsed:.0f}s is close to the 5-minute "
            "compute limit. Consider reducing --first-pass."
        )


if __name__ == "__main__":
    main()
