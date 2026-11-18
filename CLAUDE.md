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
Agent loop         (agent/agent.py)    native tool calling, no framework; computed Answer line (agent/headlines.py)
      │                                LLM client (agent/llm.py): Gemini, optional Groq failover
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
- We use the `openai` Python package only as a generic client for FREE providers' OpenAI-compatible
  endpoints. Language model: **Google AI Studio, `gemini-3.5-flash-lite`** (free tier, no card). Optional:
  **Groq, `openai/gpt-oss-120b`** (free tier) for automatic failover, plus Whisper on Groq for voice input.
  See `agent/llm.py` and `docs/eval/model_comparison.md`.
- Provider-agnostic = no vendor lock-in: both providers use the same OpenAI-compatible client code.
- No LangChain/LangGraph: 7 tools and one loop; a framework adds abstraction without value here.
- Full transparency: every tool call is logged and shown in the UI; easy to explain in the interview.
- Tradeoffs: free tiers have rate limits (requests/tokens per minute and per day) → keep tool outputs
  compact, keep the system prompt short, pace calls, fail over to Groq, show a friendly message if every
  model is out of quota. Smaller models are weaker at tool choice and explanation → mitigated because all
  numbers come from deterministic tools and the Answer line is computed by code.
- Configuration (`.env` locally, Streamlit secrets when deployed; see `.env.example` and
  `.streamlit/secrets.toml.example`): `GEMINI_API_KEY` (required), `GEMINI_MODEL` (default
  `gemini-3.5-flash-lite`), `GROQ_API_KEY` (optional: voice + failover), `GROQ_MODEL` (default
  `openai/gpt-oss-120b`). Base URLs, reasoning levels, limits and timeouts are constants in `agent/llm.py`.

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
  Weights live in `scoring/config.py` as named presets, are shown in answers, and can be overridden by the
  user with explicit numbers.

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
├── .env.example           # GEMINI_API_KEY= (required), GROQ_API_KEY= (optional), model names (optional)
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
- [x] 3. ETL → DuckDB
- [x] 4. Metrics + scoring + tests (the 4 example questions answered via Python only)
- [x] 5. Agent loop + tools (incl. guardrails: number check, grounding check, retry)
- [x] 6. Streamlit chat UI (show tool calls + sources; voice = bonus); polish done
      (empty state, follow-ups, ranking headlines, quota handling, examples in one place)
- [x] 7. DESIGN.md + README.md (+ GitHub Actions CI: offline DB build + pytest on Python 3.12)
- [ ] 8. Publish (repo is already public on GitHub, origin main; Streamlit Community Cloud deploy)  ← next

## Decisions log
One line of reasoning per decision. Values live in `scoring/config.py`; change them there, not here.

### Data
- **Window: Sep 2025 – Aug 2026 (trailing 12 months), prior window Sep 2024 – Aug 2025.** The latest complete
  T-100 months; equal-length windows make YoY growth comparable and remove seasonality.
- **Keep all downloaded months in DuckDB; windows are applied in scoring.** Lets the window move without a re-ETL.
- **Drop the carrier dimension in the processed tables.** Not needed for any KPI and keeps the parquet caches small.
- **Processed parquet caches committed in `data/processed`; raw downloads never committed.** The app builds
  offline in a few seconds with no downloads; caches are refreshed only with `python -m data.etl --from-raw`.
- **Passenger service = CLASS in (F, L) and AIRCRAFT_CONFIG != 2.** Scheduled + non-scheduled passenger
  service, excluding freighter-configured aircraft (scope: passenger airports).
- **Completion gap only on rows with departures scheduled > 0 (US carriers).** Foreign carriers report
  scheduled = 0, which made the gap negative.
- **Long-haul ≥ 3,000 statute miles, sensitivity 2,500 / 4,000; departure- and seat-weighted.** The threshold
  is an assumption, so the answer shows how much it matters.
- **NAS share = NAS delay minutes / sum of the five cause columns.** The NAS cause is the closest public
  proxy for airport/airspace capacity constraints.
- **Runway rule: commercial runways ≥ 5,000 ft (inclusive); helipads and unlabelled runways excluded.**
  Short GA strips inflated the runway count and understated operations per runway.
- **Quality report: cross-check within ±1%, the one known delay-cause flag documented.** Proves the ETL
  numbers match the sources without hiding the known discrepancy.
- **FAA NAS Status is live context only, never scored.** It is real-time and not reproducible.

### Scoring
- **Target load factor 0.85, sensitivity 0.80.** A common practical ceiling for "full"; the 0.80 case shows
  how sensitive the capacity gap is to that assumption.
- **Ranking universe = FAA large/medium/small hubs (142 airports with T-100 data).** Comparable peers;
  non-hubs distort percentiles.
- **Mid-rank percentiles against the universe; contributions add up to the score.** Deterministic, robust to
  outliers and easy to explain component by component.
- **Investment weights: demand 0.35, growth 0.25, congestion 0.25, scale 0.15.** Demand pressure is the core
  thesis; scale has the least weight so big hubs don't win by size alone.
- **Named weight presets (demand/growth/congestion/scale): balanced 0.35/0.25/0.25/0.15 (default),
  demand-led 0.50/0.20/0.15/0.15, congestion-led 0.25/0.15/0.45/0.15, growth-led 0.25/0.45/0.15/0.15,
  scale-led 0.25/0.15/0.15/0.45.** "Weight growth higher" → `rank_airports(preset="growth-led")`, so the same
  question always gives the same ranking. Custom weights only when the user gives numbers (merged into the
  preset and renormalised). The preset name and weights are shown in the Answer line and Key facts; all
  presets are used for the rank-stability check.
- **Congestion weights: delay 0.35, NAS 0.35, ops per runway 0.20, cancellations 0.10.** Ops per runway is
  a structural signal but cruder than the delay data, so it gets 20%.
- **Unmet-demand proxy weights: average LF 0.25, peak LF 0.20, growth 0.20, delay 0.10, NAS 0.15,
  completion gap 0.10.** True spilled demand is not observable, so the score is labelled a proxy.
- **The national delay rate is printed next to the airport (e.g. SFO).** Without a benchmark, a delay rate means
  little.
- **Regions and aliases are explicit data.** New England = CT, ME, MA, NH, RI, VT. "LA" resolves to LAX,
  with BUR/LGB/ONT/SNA as alternatives, and the interpretation is always stated.

### Agent
- **No framework; native tool calling with a small hand-written loop.** 7 tools and one loop; every line
  is explainable.
- **7 bundled tools, so each example question needs one tool call.** Fewer round trips mean fewer tokens and
  faster answers on free tiers.
- **Compact, readable LLM views; the full envelope goes to the UI.** Short source ids, units on every value
  (load factors as plain ratios, indices as "/ 100"), no snake_case (component labels, readable keys),
  differences as positive magnitudes with direction words ("NAS share raises SNA by 4.22 pts"), ranked
  drivers, interpretation strings, headlines and official airport names, so the LLM restates rather than
  calculates.
- **Rankings explain themselves.** Per-airport score drivers for the top 3 (weight × (percentile − 50): how much
  each component raises or lowers the score vs a median hub) and, for non-default presets, each airport's rank
  change vs the balanced preset ("BDL: 2 → 3 (down)") so a wrong premise ("why did it move up?") is corrected.
- **One tool round, then a streamed explanation.** The LLM chooses the tools (several in parallel are fine);
  as soon as they ran, the Answer line, Key facts and Evidence are shown; one more call (tool_choice "none")
  streams the bullets token by token. The checks run when the stream ends; a failed grounding check replaces
  the bullets with the retried version. A bullet cut off mid-sentence (output limit / stream ended) is dropped.
- **Memory = last 6 exchanges as text; earlier tool outputs are not resent.** Each earlier answer carries a
  one-line summary of its tool calls; answers older than the latest are truncated to 300 characters. Prompt
  size stays flat as a conversation grows (eval: turn 1 4,281 tokens, turn 5 4,626).
- **Lowest reasoning level that works (Gemini `minimal`, Groq `low`); max output 2,000 tokens for Gemini
  (its thinking counts toward the limit), 1,000 for Groq.** Fewer hidden tokens, same eval quality.
- **Pacing from the provider's rate-limit headers when sent (Groq; cached tokens don't count), else a local
  per-minute window of tokens and requests (Gemini); without failover, 429 retries are capped at 30 s.**
  Avoids most 429s; bounded waits.
- **Deterministic Answer line + LLM "Why / drivers" bullets (max 4).** Code builds the core answer from the tool
  output (`agent/headlines.py`), so rank order, names, which index is higher, shares, scores, the weight preset
  and live FAA events are always correct; the number and grounding checks run on the LLM part. "Why is X
  lower/higher?" gets its own Answer line built from the ranked contribution differences (and corrects a wrong
  premise).
- **Fast path for example cards and suggested follow-ups.** The tool and its arguments are known: the tool
  runs directly, Key facts and Evidence show in under 0.25 s, then one streamed LLM call writes the bullets.
  The call is recorded in the conversation like an LLM-chosen one and counts as the turn's tool call for the
  grounding check, so follow-ups keep their context.
- **Number traceability check across all turns.** Every number in an answer must trace to a tool output
  (envelope or LLM view); follow-ups may cite earlier turns.
- **Grounding check.** Airport codes must be grounded; the answer must not claim a tool that was not called
  (its identifier or "<name> tool"; plain topic words like "unmet demand" are not claims); rank/compare/
  compute/weight requests need a tool call unless the answer is a clear scope refusal. If the check fails,
  there is one retry with `tool_choice="required"` (fast path: same tool output, new explanation).
- **Scope: known non-US airports get a clear answer.** The 40 largest non-US destinations in the T-100 data
  (plus a few hubs) are listed in `scoring/config.py`; asking about one (code or city) returns "LHR London
  Heathrow Airport is outside the US scope" as the Answer line instead of "unknown airport".
- **Follow-up context: the last tool call and its arguments are injected into the user message.** This lets
  "rank them again…" re-call the tool with the right region. A follow-up-style question with no earlier
  result ("which is third now?") gets a note to ask a clarifying question; a ranking without scope then says
  "No previous ranking in this conversation; showing the national ranking."
- **Live FAA status is summarized in words.** Event types decoded ("WX:Wind" → "Weather: wind"), times in UTC
  and the airport's local time (state → time zone, `scoring/config.py`), raw feed codes such as "(.308)"
  removed from runway configurations, empty events never shown.
- **Refusals are helpful.** An out-of-scope answer has at most 2 sentences: what is out of scope and why, then
  one concrete in-scope offer using the question's US airports ("JFK is in scope - I can show its profile or
  compare it with other US airports"). The eval checks this on 3 out-of-scope questions.
- **Prompt rules frozen (`PROMPT_VERSION`), except the bullet sign rule and the refusal rule:** a component
  drives a score UP only if its contribution is positive; never call negative growth a driver; use "raises X" /
  "lowers X". Further tuning gave diminishing returns; the remaining limits are mitigated in the UI. The prompt
  is ~500 tokens (test cap 525).

### Models (details and numbers: `docs/eval/model_comparison.md`, eval report: `docs/eval/eval_results.md`)
- **Language model `gemini-3.5-flash-lite` (Google AI Studio free tier, thinking `minimal`).** ~250K tokens/min
  means no pacing waits; latest stable Flash-Lite in Google's models list. Eval (10 questions): 7/7 correct
  tool choices, 3/3 helpful refusals, 0 grounding failures, 0 untraceable numbers.
- **One required key, one optional.** `GEMINI_API_KEY` is required. `GROQ_API_KEY` adds voice input and
  failover to `openai/gpt-oss-120b`; without it, everything except voice works, with no failover (sidebar:
  "Voice input requires GROQ_API_KEY"). Only `GROQ_API_KEY` set → the app runs on Groq alone. Neither → the
  setup card.
- **Error policy:** Gemini 429 → Groq immediately; Gemini 5xx, timeout (no first token within 8 s; a stream
  that has started is never cut) or connection error → one retry
  after 2 s, then Groq; Gemini would wait > 5 s for its rate limit → Groq. Once on Groq, the rest of the turn
  stays there. The "Done" line names the model that answered (and the failover reason). Free-tier servers do
  return 5xx (seen in the evals), so one retry recovers most of them without switching models.
- **Daily quota 429 → that model is skipped for the session; all exhausted → a clear message.**
  Key facts, Evidence and cached examples still work.
- **Not used: `qwen/qwen3.8-27b` (malformed tool calls) and `llama-3.3-70b-versatile` (not available on the
  evaluation key).**
- **Known LLM limits:** direction wording and weak explanations in the bullets. The core answer cannot be
  misquoted (computed Answer line); Key facts and Evidence are deterministic.

### UI
- **Streamlit with Altair charts.** Key facts and Evidence are built from the tool outputs, not from the LLM
  text, so the numbers on screen are deterministic.
- **First-question answer cache only.** It is keyed by question, model, prompt version and data build date
  (a new `PROMPT_VERSION` invalidates it; it lives in memory, so a restart clears it), and stored only when an
  answer has no error, needed no retry and passed every check. The demo questions answer instantly without
  spending quota.
- **Progressive answers.** Answer line, Key facts and Evidence appear as soon as the tool ran (card click
  under 0.2 s; typed question typically 1-2 s); the status line says "Facts ready · writing explanation…" and a small
  spinner sits where the bullets will appear; the bullets then stream in.
- **Each element is rendered once.** Every answer uses one fixed frame of slots (status, Answer, Why / drivers,
  Key facts + Evidence, follow-ups) live and in history, so a rerun replaces elements in place. The welcome
  cards sit in one placeholder that is emptied as soon as a question is submitted from any path (typed, card,
  follow-up, sidebar, voice); follow-up buttons appear only under the latest answer and not while a new answer
  loads.
- **Example questions appear in one place.** They are cards on the empty state and in the sidebar once a
  conversation exists, so they are never shown twice.
- **Follow-up suggestions are deterministic.** They are based on the last successful tool call, with at
  most 3 and none after errors, and never repeat a question already asked in the conversation (backup
  suggestions fill the slot). They are guaranteed to be answerable.
- **Human-readable everywhere.** Key facts and the LLM's input use labels, not snake_case; magnitudes are
  positive with direction words; the "Shown but not ranked" list stops at 5 ("and N more").
- **Light theme with a viewer toolbar.** It is a clean demo for analysts.
- **Voice is beta and needs `GROQ_API_KEY`.** It uses Whisper on Groq with a domain vocabulary prompt (unmet
  demand, load factor, congestion, New England, terminal expansion, airport codes); the transcript can be
  edited before sending and is cleared after Send.
- **The About panel and sidebar describe the active setup** (language model, failover on/off, voice on/off),
  generated from the configuration so they cannot go stale.
- **The DuckDB file is built automatically (offline) on first run, and the keys are read from `.env` or
  `st.secrets`.** This lets the app run with zero setup locally and on Streamlit Community Cloud.

### Working rules
- **No scripted edits (Python/PowerShell rewriting files); use the normal edit tool only, and
  `.gitattributes` sets `eol=lf`.** Scripted edits broke encoding and line endings.
- **No real-LLM runs (eval_agent.py, measure_fast_path.py, chat_cli) unless explicitly asked; use the FakeLLM
  tests.** This protects the free daily quotas.
- **Never commit secrets (`.env`, `.streamlit/secrets.toml`), and never modify the user's `.env`.** The repo is
  public.
- **Plan first ("PLAN ONLY"), implement after approval.** Before every commit, run `pytest -q`, then restart
  the app, then commit with the given message and push.
