# Airport Investment Intelligence Agent — Project Guide

Take-home assignment for Deloitte Digital (Forward Deployed Engineer).
The repo will be **public** on GitHub. Never commit secrets.

## Goal
An AI agent that helps investment analysts find US airports where terminal/capacity
renovations are most likely to pay off, based on flight and passenger demand vs. capacity.

Example questions it MUST answer well:
1. Which airports in New England are strong candidates for terminal expansion?
2. Compare LA (LAX) and Santa Ana (SNA) airport congestion levels.
3. What is the percentage of long-haul flights out of Anchorage (ANC)?
4. What is the unmet flight demand at SFO and why?

## Hard requirements (from the assignment)
- Use public APIs for airport/aviation data
- Deterministic scoring / ranking logic (NOT only LLM output)
- Chat interface (voice = bonus)
- Explain reasoning clearly; support follow-up questions
- Clearly communicate assumptions, uncertainty and scope
- Deliverables: source code + short DESIGN.md (scoring methodology, key tradeoffs, where/how AI is used)

## Core design principle (non-negotiable)
**The LLM never produces a number.** Every metric, score and ranking is computed by
deterministic Python tools. The LLM only: (a) understands the question, (b) picks tools
and arguments, (c) explains tool output in plain language, including assumptions and caveats.
Every tool result must carry its own `sources`, `data_period` and `assumptions` fields so the
LLM can cite them.

## Architecture
```
Streamlit chat UI  (app.py)
      │
Agent loop         (agent/agent.py)    OpenAI-compatible client, native tool calling, no framework
      │
Tools              (agent/tools.py)    thin wrappers -> JSON-serializable dicts
      │
Scoring / metrics  (scoring/*.py)      pure functions, unit tested, no I/O
      │
Data layer         (data/*.py)         ETL -> local DuckDB file (data/airports.duckdb)
      │
Public sources     BTS (data.bts.gov SODA API / TranStats), FAA, OurAirports, FAA NAS Status
```

### Harness choice — free LLM via an OpenAI-compatible API + our own small tool loop
Constraint: **zero cost**. No paid services anywhere in the project.
- We use the `openai` Python package only as a generic client, pointed at a FREE provider via
  `LLM_BASE_URL`. Default: **Groq free tier** (no credit card), model set by `LLM_MODEL`
  (e.g. `openai/gpt-oss-120b` or `llama-3.3-70b-versatile`).
- Same code works with Google AI Studio (Gemini, free tier, OpenAI-compatible endpoint) or a
  local Ollama model — just change 3 env vars. Provider-agnostic = no vendor lock-in.
- No LangChain/LangGraph: ~5 tools and one loop; a framework adds abstraction without value here.
- Full transparency: every tool call is logged and shown in the UI; easy to explain in the interview.
- Tradeoffs: free tiers have low rate limits (requests/tokens per minute) → keep tool outputs
  compact, keep the system prompt short, retry with backoff on HTTP 429, show a friendly
  message if the limit is hit. Open models are somewhat weaker than frontier models at tool
  choice → this is mitigated because all numbers come from deterministic tools anyway.
- Env vars: `LLM_API_KEY`, `LLM_BASE_URL` (default `https://api.groq.com/openai/v1`), `LLM_MODEL`.

## Data sources (verify each in the data spike before building on it)
| Source | Use | Access |
|---|---|---|
| BTS T-100 Segment (All Carriers, domestic + international) | passengers, seats, departures (scheduled/performed), distance → load factor, long-haul share, growth | data.bts.gov Socrata API if the needed table exists; fallback: TranStats CSV download (documented, committed under data/raw) |
| BTS Airline On-Time / Delay Cause by airport (monthly) | arrival delay rate, delay minutes by cause (NAS share = capacity/ATC signal) | TranStats download / data.transportation.gov Socrata |
| FAA Passenger Boarding (Enplanement) data CY2024/CY2025 | official enplanements, hub size, YoY growth | FAA Excel files |
| OurAirports (airports.csv, runways.csv) | metadata: name, state, lat/lon, runway count | public CSV |
| FAA NAS Status API (nasstatus.faa.gov) | LIVE delays / ground stops — context only, not used in scoring | live HTTP call at query time |

Rules:
- ETL scripts are re-runnable: `python -m data.etl` rebuilds the DuckDB from sources.
- Store raw downloads in `data/raw/` with a README noting source URL + download date.
- Scope: US commercial passenger airports. Passenger service only (exclude all-cargo flights) unless stated.

## KPIs (v1 — refine after data spike)
All computed per airport over a trailing 12-month window (state the window in every answer).
- **Load factor (LF)** = passengers / seats on departing segments. High LF → demand near capacity.
- **Passenger growth** = YoY % change in enplanements.
- **Delay rate** = arrivals delayed ≥15 min / arrivals.
- **NAS delay share** = NAS delay minutes / total delay minutes (proxy for airport/airspace capacity constraint).
- **Completion gap** = 1 − departures performed / departures scheduled.
- **Long-haul share** = departures with segment distance ≥ 3,000 statute miles / all departures.
  Threshold is an ASSUMPTION, configurable; also report the seat-weighted share and a sensitivity
  table (e.g. 2,500 / 3,000 / 4,000 mi).
- **Unmet demand proxy** = composite of LF, delay/NAS share, growth and completion gap.
  True unmet (spilled) demand is NOT directly observable — always say so.
- **Investment Score (0–100)** = weighted sum of percentile-normalized components:
  demand pressure (LF), growth, congestion (delay rate + NAS share), scale (enplanements).
  Weights live in `scoring/config.py`, are shown in answers, and can be overridden by the user.

## Repo layout
```
.
├── app.py                 # Streamlit chat UI
├── agent/
│   ├── agent.py           # tool-use loop + conversation memory
│   ├── tools.py           # tool definitions (JSON schema) + dispatch
│   └── prompts.py         # system prompt
├── scoring/
│   ├── config.py          # weights, thresholds, region definitions
│   ├── metrics.py         # KPI functions
│   └── ranking.py         # normalization + composite score
├── data/
│   ├── etl.py             # fetch + clean + load DuckDB
│   ├── sources/           # one module per source
│   └── raw/               # downloaded files (+ README with provenance)
├── tests/                 # pytest: scoring logic + the 4 example questions
├── DESIGN.md
├── README.md
├── requirements.txt
├── .env.example           # LLM_API_KEY=, LLM_BASE_URL=, LLM_MODEL=
└── .gitignore             # .env, data/*.duckdb, __pycache__, .venv
```

## Conventions
- Python 3.11+ (author has 3.14 installed; if a dependency has no 3.14 wheel, tell the author
  and suggest a 3.12 venv rather than hacking around it). Type hints, small pure functions in scoring/.
- All data sources and the LLM must be free. Never add a paid API or SDK.
- Region definitions are explicit data (e.g. New England = CT, ME, MA, NH, RI, VT).
- City → airport mapping is explicit; ambiguous names (e.g. "LA") are resolved and stated
  ("LA interpreted as LAX").
- Every tool returns: `{"data": ..., "sources": [...], "data_period": "...", "assumptions": [...], "caveats": [...]}`.
- Tests for every metric with small hand-checked fixtures.
- Keep it simple and readable over clever — the author must explain every line in an interview.

## Build plan (status)
- [x] 1. Environment + repo skeleton
- [x] 2. Data spike: verify each source, record exact endpoints/fields in data/raw/README.md
- [ ] 3. ETL → DuckDB
- [ ] 4. Metrics + scoring + tests (the 4 example questions answered via Python only)
- [ ] 5. Agent loop + tools
- [ ] 6. Streamlit chat UI (show tool calls + sources; voice = bonus)
- [ ] 7. DESIGN.md + README.md
- [ ] 8. Publish public GitHub repo
