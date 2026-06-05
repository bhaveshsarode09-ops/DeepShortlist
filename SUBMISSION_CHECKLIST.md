# Submission Checklist

## Files to Submit on Portal
- [ ] team_xxx.csv (rename to your participant ID)
- [ ] GitHub repo URL: https://github.com/bhaveshsarode09-ops/DeepShortlist
- [ ] Colab sandbox link: [fill after deploying]
- [ ] AI tools declared: Claude
- [ ] Methodology summary: from submission_metadata.yaml

## Pre-Submit Validation
- [ ] python validate_submission.py team_xxx.csv → "Submission is valid"
- [ ] Exactly 100 rows in CSV
- [ ] Ranks 1-100 all present exactly once
- [ ] Scores non-increasing
- [ ] No duplicate candidate_ids
- [ ] Colab notebook runs end-to-end without errors
- [ ] GitHub repo is public
- [ ] README.md is complete
- [ ] submission_metadata.yaml is filled

## Things to Remember for Stage 5 (Defend Your Work)
- Know the 4-stage pipeline by heart
- Know why availability weight is 20% (JD explicitly says this)
- Know the 6 honeypot detection rules
- Know why FAISS first-pass is used (speed)
- Know the penalty multipliers and why each exists
- Know how reasoning varies by rank tier
