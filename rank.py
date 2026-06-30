#!/usr/bin/env python3
"""
rank.py — Redrob Hackathon: Intelligent Candidate Discovery & Ranking Challenge

Produces a top-100 ranked CSV (and XLSX) of candidates for the released
Senior AI Engineer JD from a 100K-candidate pool.

Design choices (explained in the pitch deck / README too):
  - No GPU, no hosted LLM calls during ranking (constraint: 5 min / 16GB / CPU-only).
  - Score = explainable weighted combination of:
        1. Hard-skill match (production embeddings/retrieval, vector DB / hybrid
           search, Python, ranking-evaluation experience) read from the FULL
           text of the profile (headline, summary, every career_history
           description, skills) — not just the skills list or job title.
           This is what lets us catch "Tier 5 in plain language" candidates
           and reject "keyword stuffers" whose skills list is full of AI terms
           but whose actual work never touches them.
        2. Nice-to-have skill match (fine-tuning, learning-to-rank, HR-tech,
           distributed systems, open source).
        3. Experience-band fit around the JD's 5-9 (sweet spot 6-8) years.
        4. Location / relocation fit against the JD's stated geography.
        5. JD-specific disqualifier penalties (pure research w/ no production,
           recent LangChain-only "AI experience", architects who haven't
           coded in 18 months, consulting-only career, CV/speech/robotics
           without NLP/IR, title-chasers).
        6. A behavioral-availability multiplier built from redrob_signals
           (recency of activity, recruiter response rate, interview
           completion rate, notice period, open_to_work flag, profile
           completeness) — a perfect-on-paper but inactive/unresponsive
           candidate is down-weighted, per the JD's explicit instruction.
        7. A data-quality / honeypot penalty: candidates whose profile
           contains internally-impossible claims (e.g. "expert" proficiency
           in a skill with 0 months of use, or stated total experience wildly
           inconsistent with the sum of their career-history durations) are
           pushed to the bottom rather than excluded outright, so the system
           is robust even if our heuristic catches something that isn't
           actually a deliberately planted honeypot.

  - Single streaming pass over candidates.jsonl, O(1) memory per candidate,
    a small max-heap holds the current top 100 -> runs in a couple of
    minutes on 100K rows on a laptop CPU.
"""

import argparse
import csv
import heapq
import json
import re
import sys
import time
from datetime import date, datetime

# --------------------------------------------------------------------------
# Reference date the synthetic dataset is anchored to (i.e. "today" inside
# this dataset's world) — used for activity-recency calculations.
# Derived from the max last_active_date / signup_date we'd expect in a
# dataset generated around the hackathon window.
# --------------------------------------------------------------------------
DATASET_TODAY = date(2026, 6, 1)


# --------------------------------------------------------------------------
# JD-derived keyword groups. Kept as simple word/phrase lists (not a model)
# so the whole thing is auditable, fast, and explainable in the reasoning
# column — which the spec explicitly rewards.
# --------------------------------------------------------------------------
MUST_HAVE = {
    "embeddings_retrieval": [
        "sentence-transformers", "sentence transformers", "embedding",
        "openai embedding", "bge embedding", "e5 embedding", "dense retrieval",
        "semantic search", "retrieval-augmented", "rag pipeline", "rag system",
    ],
    "vector_db_hybrid_search": [
        "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
        "elasticsearch", "faiss", "vector database", "vector store",
        "hybrid search", "hybrid retrieval", "bm25",
    ],
    "python": [
        "python",
    ],
    "eval_framework": [
        "ndcg", "mrr", "map@", "mean average precision", "precision@",
        "offline-to-online", "offline to online", "a/b test", "ab test",
        "evaluation framework", "ranking evaluation", "online evaluation",
    ],
}

NICE_TO_HAVE = {
    "fine_tuning": ["lora", "qlora", "peft", "fine-tun", "finetun"],
    "learning_to_rank": ["learning-to-rank", "learning to rank", "xgboost",
                          "neural ranking", "ltr model"],
    "hr_tech": ["hr-tech", "hr tech", "recruiting tech", "recruitment platform",
                "talent marketplace", "marketplace product"],
    "distributed_scale": ["distributed system", "large-scale inference",
                           "inference optimization", "high-scale", "at scale"],
    "open_source": ["open-source", "open source contribution", "published paper",
                     "conference talk", "oss maintainer"],
}

PRODUCTION_SIGNAL = [
    "production", "deployed", "shipped", "real users", "real-world",
    "live system", "scale", "end-to-end", "end to end",
]

PRE_LLM_ML_SIGNAL = [
    "machine learning", "recommendation system", "recommender system",
    "search ranking", "information retrieval", "nlp", "ranking system",
    "click-through", "ctr prediction", "collaborative filtering",
]

LANGCHAIN_OPENAI_ONLY = ["langchain"]

ARCHITECT_TITLES = ["architect", "tech lead", "technical lead",
                     "engineering manager", "head of", "director"]
CODING_SIGNAL = ["implemented", "built", "wrote", "coded", "hands-on",
                  "hands on", "shipped code", "developed", "designed and built"]

CONSULTING_FIRMS = ["tcs", "tata consultancy", "infosys", "wipro", "accenture",
                     "cognizant", "capgemini"]

CV_SPEECH_ROBOTICS = ["computer vision", "image classification", "object detection",
                       "speech recognition", "robotics", "autonomous driving",
                       "self-driving"]
NLP_IR_SIGNAL = ["nlp", "natural language processing", "information retrieval",
                  "text classification", "named entity", "embedding", "retrieval",
                  "ranking", "search"]

RESEARCH_ONLY_SIGNAL = ["research scientist", "research lab", "phd", "academia",
                          "university research", "postdoc", "research fellow"]

TIER1_PRIMARY = {"pune", "noida"}
TIER1_WELCOME = {"hyderabad", "mumbai", "delhi", "gurgaon", "gurugram",
                  "new delhi", "noida", "pune", "bangalore", "bengaluru"}


def to_lower_blob(c):
    """Concatenate every free-text field into one lowercase blob for keyword
    search. We deliberately look beyond the skills list / title, since the
    JD explicitly says the right signal lives in career history, not just
    declared skills."""
    parts = [
        c["profile"].get("headline", ""),
        c["profile"].get("summary", ""),
        c["profile"].get("current_title", ""),
        c["profile"].get("current_industry", ""),
    ]
    for ch in c.get("career_history", []):
        parts.append(ch.get("title", ""))
        parts.append(ch.get("description", ""))
        parts.append(ch.get("industry", ""))
    for s in c.get("skills", []):
        parts.append(s.get("name", ""))
    return " | ".join(p for p in parts if p).lower()


def any_kw(blob, kws):
    return any(k in blob for k in kws)


def matched_kws(blob, kws):
    return [k for k in kws if k in blob]


def current_role_blob(c):
    cur = next((ch for ch in c.get("career_history", []) if ch.get("is_current")), None)
    if not cur:
        return "", 0
    return (cur.get("title", "") + " " + cur.get("description", "")).lower(), \
        cur.get("duration_months", 0)


def parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


def score_candidate(c):
    """Returns (score: float 0..1, reasoning: str, honeypot_flag: bool)."""
    blob = to_lower_blob(c)
    profile = c["profile"]
    signals = c["redrob_signals"]
    reasons_positive = []
    reasons_negative = []

    # ---------- 1. Must-have skill match ----------
    must_hit_groups = 0
    must_hit_names = []
    for group, kws in MUST_HAVE.items():
        hits = matched_kws(blob, kws)
        if hits:
            must_hit_groups += 1
            must_hit_names.append(group)
    must_have_score = must_hit_groups / len(MUST_HAVE)  # 0..1

    # ---------- 2. Nice-to-have ----------
    nice_hit_groups = 0
    nice_hit_names = []
    for group, kws in NICE_TO_HAVE.items():
        hits = matched_kws(blob, kws)
        if hits:
            nice_hit_groups += 1
            nice_hit_names.append(group)
    nice_have_score = nice_hit_groups / len(NICE_TO_HAVE)

    # ---------- 3. Experience band fit (sweet spot 6-8, JD range 5-9) ----------
    yoe = profile.get("years_of_experience", 0) or 0
    if 6 <= yoe <= 8:
        exp_score = 1.0
    elif 5 <= yoe <= 9:
        exp_score = 0.85
    elif 3 <= yoe < 5 or 9 < yoe <= 11:
        exp_score = 0.5
    else:
        exp_score = 0.2

    # ---------- 4. Location / relocation fit ----------
    loc = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower()
    willing = signals.get("willing_to_relocate", False)
    if country == "india":
        if any(k in loc for k in TIER1_PRIMARY):
            loc_score = 1.0
        elif any(k in loc for k in TIER1_WELCOME):
            loc_score = 0.85
        elif willing:
            loc_score = 0.55
        else:
            loc_score = 0.35
    else:
        loc_score = 0.35 if willing else 0.1

    # ---------- 5. Disqualifier / penalty multipliers ----------
    penalty = 1.0
    cur_blob, cur_months = current_role_blob(c)

    # 5a. Pure research / academia, no production deployment anywhere
    if any_kw(blob, RESEARCH_ONLY_SIGNAL) and not any_kw(blob, PRODUCTION_SIGNAL):
        penalty *= 0.15
        reasons_negative.append("research-only background with no visible production deployment")

    # 5b. AI experience = recent LangChain/OpenAI-only, no pre-LLM ML/IR background
    if any_kw(cur_blob, LANGCHAIN_OPENAI_ONLY) and cur_months < 12 and not any_kw(blob, PRE_LLM_ML_SIGNAL):
        penalty *= 0.3
        reasons_negative.append("AI experience looks limited to recent LangChain work with no earlier ML/IR background")

    # 5c. Architect/manager title, long tenure, no coding signal in current role
    if any(k in (profile.get("current_title") or "").lower() for k in ARCHITECT_TITLES) \
            and cur_months >= 18 and not any_kw(cur_blob, CODING_SIGNAL):
        penalty *= 0.4
        reasons_negative.append("current role reads as architecture/management with no recent hands-on coding signal")

    # 5d. Consulting-only career (no product company experience at all)
    all_companies = [(ch.get("company") or "").lower() for ch in c.get("career_history", [])]
    if all_companies and all(any(f in comp for f in CONSULTING_FIRMS) for comp in all_companies):
        penalty *= 0.3
        reasons_negative.append("entire career at consulting firms with no product-company experience")

    # 5e. CV / speech / robotics background without NLP/IR exposure
    if any_kw(blob, CV_SPEECH_ROBOTICS) and not any_kw(blob, NLP_IR_SIGNAL):
        penalty *= 0.4
        reasons_negative.append("background is computer vision/speech/robotics without NLP or retrieval exposure")

    # 5f. Title-chasing: 3+ short (<18mo) stints with escalating seniority
    short_stints = sum(1 for ch in c.get("career_history", []) if (ch.get("duration_months") or 0) < 18)
    if short_stints >= 3 and len(c.get("career_history", [])) >= 3:
        penalty *= 0.7
        reasons_negative.append("career pattern shows several short (<18mo) stints, a title-chasing signal the JD flags")

    # ---------- 6. Behavioral / availability multiplier ----------
    behavior = 1.0
    last_active = parse_date(signals.get("last_active_date"))
    if last_active:
        days_inactive = (DATASET_TODAY - last_active).days
        if days_inactive <= 30:
            behavior *= 1.10
        elif days_inactive <= 90:
            behavior *= 1.0
        elif days_inactive <= 180:
            behavior *= 0.7
            reasons_negative.append(f"inactive on the platform for ~{days_inactive} days")
        else:
            behavior *= 0.4
            reasons_negative.append(f"inactive on the platform for ~{days_inactive} days")

    if signals.get("open_to_work_flag"):
        behavior *= 1.08
    else:
        behavior *= 0.85

    rrr = signals.get("recruiter_response_rate", 0) or 0
    behavior *= (0.7 + 0.6 * rrr)  # 0.7..1.3

    icr = signals.get("interview_completion_rate", 0) or 0
    behavior *= (0.85 + 0.3 * icr)  # 0.85..1.15

    notice = signals.get("notice_period_days", 60) or 0
    if notice <= 30:
        behavior *= 1.1
    elif notice <= 60:
        behavior *= 1.0
    else:
        behavior *= 0.9

    pcs = signals.get("profile_completeness_score", 50) or 50
    behavior *= (0.85 + 0.3 * (pcs / 100))

    if signals.get("recruiter_response_rate", 0) and signals.get("recruiter_response_rate") <= 0.10 \
            and last_active and (DATASET_TODAY - last_active).days > 150:
        reasons_negative.append("low recruiter response rate combined with long inactivity suggests not actually available")

    # ---------- 7. Honeypot / data-quality flag ----------
    honeypot = False
    total_months = sum((ch.get("duration_months") or 0) for ch in c.get("career_history", []))
    if yoe and total_months:
        if abs((total_months / 12.0) - yoe) > 3.0:
            honeypot = True
    for s in c.get("skills", []):
        if s.get("proficiency") == "expert" and (s.get("duration_months") or 0) == 0:
            honeypot = True
            break
    if honeypot:
        penalty *= 0.02
        reasons_negative.append("profile contains internally inconsistent claims (likely a data-quality / honeypot profile)")

    # ---------- Combine ----------
    base = (0.45 * must_have_score
            + 0.15 * nice_have_score
            + 0.15 * exp_score
            + 0.15 * loc_score
            + 0.10 * (1.0 if any_kw(blob, PRODUCTION_SIGNAL) else 0.0))

    # Note: deliberately NOT clipped to 1.0 here. Behavior/penalty multipliers
    # can push the raw composite above 1.0 or below 0.0; we keep the raw value
    # for ranking precision and rescale to a clean 0-1 band only after the
    # top 100 are selected (see main()), so scores stay differentiated instead
    # of bunching at a hard ceiling.
    score = max(0.0, base * penalty * behavior)

    # ---------- Reasoning ----------
    if must_hit_names:
        reasons_positive.append(f"matches on {', '.join(must_hit_names)}")
    if nice_hit_names:
        reasons_positive.append(f"also shows {', '.join(nice_hit_names[:2])}")
    reasons_positive.append(f"{yoe} yrs experience, based in {profile.get('location', 'unknown')}")

    pos = "; ".join(reasons_positive)
    neg = ("; concerns: " + "; ".join(reasons_negative)) if reasons_negative else ""
    reasoning = (pos + neg).strip()
    if len(reasoning) > 280:
        reasoning = reasoning[:277] + "..."

    return score, reasoning, honeypot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="path to candidates.jsonl (or .jsonl.gz)")
    ap.add_argument("--out", required=True, help="output CSV path")
    args = ap.parse_args()

    t0 = time.time()
    opener = open
    path = args.candidates
    if path.endswith(".gz"):
        import gzip
        opener = gzip.open

    heap = []  # min-heap of (score, -negindex, candidate_id, rank_tuple) keep top 100
    counter = 0
    n_total = 0
    n_honeypot = 0

    def iter_candidates():
        if path.endswith(".json") and not path.endswith(".jsonl"):
            # Plain JSON array (used for the small sandbox sample)
            with open(path, "rt", encoding="utf-8") as jf:
                arr = json.load(jf)
            for c in arr:
                yield c
        else:
            with opener(path, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue

    for c in iter_candidates():
        n_total += 1
        score, reasoning, honeypot = score_candidate(c)
        if honeypot:
            n_honeypot += 1
        counter += 1
        item = (score, c["candidate_id"], reasoning)
        if len(heap) < 100:
            heapq.heappush(heap, item)
        else:
            if item > heap[0]:
                heapq.heapreplace(heap, item)

    top = sorted(heap, key=lambda x: (-x[0], x[1]))

    # Rescale raw composite scores to a clean, strictly-differentiated band
    # (0.40 - 0.99) so rank 1 and rank 100 are clearly distinguished and no
    # two candidates show an identical score unless their raw composites were
    # genuinely tied.
    raw_scores = [t[0] for t in top]
    smax, smin = max(raw_scores), min(raw_scores)
    spread = smax - smin
    rescaled = []
    for score, cid, reasoning in top:
        if spread > 1e-9:
            r = 0.40 + 0.59 * ((score - smin) / spread)
        else:
            r = 0.70
        rescaled.append((round(r, 4), cid, reasoning))

    # Rounding to 4 decimals can occasionally make two distinct raw scores
    # collide. Re-sort by (rounded_score desc, candidate_id asc) so any tie
    # group is internally ordered by ascending candidate_id, per spec
    # section 3's tie-break rule, before assigning final ranks.
    rescaled.sort(key=lambda x: (-x[0], x[1]))

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, (r, cid, reasoning) in enumerate(rescaled, start=1):
            w.writerow([cid, i, f"{r:.4f}", reasoning])

    elapsed = time.time() - t0
    print(f"Processed {n_total} candidates ({n_honeypot} flagged as honeypot/data-quality issues) "
          f"in {elapsed:.1f}s. Wrote top {len(top)} to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
