#!/usr/bin/env python3
"""
rank.py — DeepShortlist v2
Single command entry point.

Usage:
  python rank.py --candidates candidates.jsonl.gz --out team_xxx.csv

No model downloads. No internet. No GPU. Pure sklearn + numpy.
Requirements: pip install scikit-learn scipy pandas numpy
"""
from __future__ import annotations
import argparse, gzip, json, logging, sys, time
from pathlib import Path
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S", stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def load_candidates(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        logger.error(f"File not found: {path}"); sys.exit(1)
    logger.info(f"Loading {p.name} ({p.stat().st_size/1e6:.1f} MB)…")
    t0 = time.time(); cands = []
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(str(p), "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try: cands.append(json.loads(line))
                except json.JSONDecodeError: pass
    if not cands:   # try JSON array format
        with open(str(p), encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list): cands = data
    logger.info(f"Loaded {len(cands):,} in {time.time()-t0:.1f}s")
    return cands


def validate(df: pd.DataFrame, out: str) -> bool:
    ok = True
    if len(df) != 100:
        logger.warning(f"Rows={len(df)} (expected 100)")
    for col in ["candidate_id","rank","score","reasoning"]:
        if col not in df.columns:
            logger.error(f"Missing column: {col}"); ok = False
    if df["candidate_id"].duplicated().any():
        logger.error("Duplicate candidate_ids"); ok = False
    scores = df["score"].values
    for i in range(len(scores)-1):
        if scores[i] < scores[i+1]:
            logger.error(f"Score inversion at ranks {i+1}-{i+2}"); ok = False; break
    return ok


def main():
    ap = argparse.ArgumentParser(description="DeepShortlist v2 — Offline Candidate Ranking")
    ap.add_argument("--candidates", "-c", required=True)
    ap.add_argument("--out",        "-o", required=True)
    ap.add_argument("--top-k",   type=int, default=100)
    ap.add_argument("--first-pass", type=int, default=3000)
    ap.add_argument("--jd",  default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.verbose: logging.getLogger().setLevel(logging.DEBUG)

    logger.info("="*60)
    logger.info("  DeepShortlist v2 | BM25+ · LSA · RRF Ensemble")
    logger.info("  Offline · No LLM · No GPU · Pure Algorithms")
    logger.info("="*60)

    jd_text = None
    if args.jd:
        p = Path(args.jd)
        if p.exists(): jd_text = p.read_text(encoding="utf-8")

    from src.ranker import DeepShortlistRanker
    cands  = load_candidates(args.candidates)
    ranker = DeepShortlistRanker(first_pass=args.first_pass, jd_text=jd_text)
    df     = ranker.rank(cands, top_k=args.top_k)

    if validate(df, args.out):
        logger.info("Validation passed ✓")
    else:
        logger.warning("Validation issues — check before submitting")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    logger.info(f"Saved: {args.out}")
    logger.info(f"Score spread: {df['score'].max()-df['score'].min():.4f}")
    logger.info("Top 10:")
    for _, r in df.head(10).iterrows():
        logger.info(f"  #{int(r['rank']):>3} {r['candidate_id']}  score={r['score']:.4f}  {str(r['reasoning'])[:70]}…")


if __name__ == "__main__":
    main()
