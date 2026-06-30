# Redrob Hackathon — Intelligent Candidate Discovery & Ranking Challenge

Ranks the top 100 candidates (out of a 100,000-candidate pool) against
Redrob's released Senior AI Engineer — Founding Team job description.

## Quick start

```bash
pip install -r requirements.txt   # only needed for the optional xlsx export
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
python csv_to_xlsx.py --csv ./submission.csv --out ./submission.xlsx   # optional
```

That single `rank.py` command is the full reproduction path: it streams
`candidates.jsonl`, scores every candidate, and writes the top-100 ranked
CSV. No other steps, no pre-computation, no manual edits.

- Runtime: ~23 seconds on 100,000 candidates on a standard CPU laptop (well
  under the 5-minute budget).
- Memory: streams the file line by line; never holds more than ~100 scored
  candidates plus the current line in memory.
- No GPU. No network calls. No hosted LLM calls during ranking.

`rank.py` also accepts the small `sample_data/sample_candidates.json` array
(used by the sandbox notebook for a quick end-to-end check on ~50
candidates):

```bash
python rank.py --candidates sample_data/sample_candidates.json --out demo_out.csv
```

## Why a rule-based scorer, not an LLM-per-candidate approach

The spec is explicit that an LLM call per candidate cannot fit a 100K-row
pool inside a 5-minute CPU budget. So the ranker is a transparent, weighted
scoring function over features pulled from the full text of each profile —
fast, auditable, and (importantly for Stage 4 reasoning review) it produces
a reasoning string built directly from the same signals that drove the
score, so there's no separate "explain it after the fact" step that could
hallucinate.

## How the score is built

For each candidate:

1. **Hard-skill match (45%)** — keyword/phrase search across the *entire*
   profile text (headline, summary, every career-history description, and
   the skills list — not just the title or the skills array) for four
   JD "absolutely need" groups: production embeddings/retrieval, vector
   database / hybrid search, Python, and ranking-evaluation experience
   (NDCG/MRR/MAP/A-B testing). Searching the full text, not just declared
   skills, is what catches "Tier 5 in plain language" candidates and
   rejects keyword-stuffers whose skills list is AI-flavored but whose
   actual work never touches it.
2. **Nice-to-have match (15%)** — fine-tuning (LoRA/QLoRA/PEFT),
   learning-to-rank, HR-tech background, distributed/scale experience,
   open-source signal.
3. **Experience-band fit (15%)** — peaks at 6-8 years (JD's stated sweet
   spot), full credit through 5-9, partial credit outside that with a
   smooth falloff rather than a hard cutoff (the JD itself says the range
   is "not a requirement").
4. **Location / relocation fit (15%)** — Pune/Noida score highest (JD's
   stated preference), other Tier-1 cities the JD explicitly welcomes next,
   then relocation-willing candidates elsewhere in India, then
   relocation-willing candidates outside India last (JD: no visa
   sponsorship, case-by-case).
5. **Production-shipping signal (10%)** — presence of "deployed / shipped /
   real users / at scale / end-to-end" language anywhere in career history.

Then JD-specific **disqualifier penalties** are applied multiplicatively
(each one is a direct read of a paragraph in `job_description.docx`'s
"things we explicitly do NOT want" section):

- Pure research/academia background with no production-deployment language
  anywhere → heavy penalty.
- "AI experience" limited to a recent (<12mo) LangChain-only role with no
  earlier ML/IR background → heavy penalty.
- Architect/tech-lead/manager title held 18+ months with no hands-on-coding
  language in that role's description → penalty.
- Entire career at the named consulting firms (TCS, Infosys, Wipro,
  Accenture, Cognizant, Capgemini) with zero product-company experience →
  penalty.
- Computer vision / speech / robotics background with no NLP/IR exposure →
  penalty.
- Three or more sub-18-month stints (title-chasing pattern) → penalty.

Then a **behavioral-availability multiplier** from `redrob_signals` (per
`redrob_signals_doc.docx`: a perfect-on-paper but inactive/unresponsive
candidate isn't actually hireable): recency of `last_active_date`,
`open_to_work_flag`, `recruiter_response_rate`, `interview_completion_rate`,
`notice_period_days`, and `profile_completeness_score`.

Finally a **data-quality / honeypot penalty**: profiles with internally
inconsistent claims (declared years-of-experience wildly mismatched against
the sum of career-history durations, or "expert" proficiency in a skill
listed with 0 months of use) are pushed to the bottom rather than hard
excluded, so the system degrades gracefully even where the heuristic
over-fires. On the full 100K pool this flags 68 candidates — close to the
documented ~80 planted honeypots — and **0 of them land in the final top
100** (well under the 10% disqualification threshold).

Raw composite scores are rescaled to a 0.40–0.99 band only *after* the top
100 are selected, so the `score` column stays meaningfully differentiated
between rank 1 and rank 100 instead of bunching near a hard ceiling.

## Repo layout

```
rank.py                 # the full ranking pipeline (stdlib only)
csv_to_xlsx.py           # optional CSV -> XLSX converter for the portal
requirements.txt
sample_data/
  sample_candidates.json # 50-candidate sample, used by the sandbox notebook
sandbox/
  sandbox_demo.ipynb     # Colab notebook: runs rank.py end-to-end on the sample
submission_metadata.yaml # mirrors the portal submission form
README.md
```

## Sandbox

See `sandbox/sandbox_demo.ipynb` — open in Google Colab, run all cells.
It runs `rank.py` against the bundled 50-candidate sample and produces a
ranked CSV in well under a minute, on CPU, no network calls. This is the
quick reproducibility check; the full 100K-candidate run is reproduced from
this same `rank.py` via the command above.

## AI tools used

See `submission_metadata.yaml` for the declaration.
