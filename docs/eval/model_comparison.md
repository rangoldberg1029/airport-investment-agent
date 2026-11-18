# Model comparison and choice

All runs use the same agent, tools, deterministic scoring and question script (`python scripts/eval_agent.py`):
the 4 example questions plus 3 follow-ups in one conversation. Only free tiers were considered.
The final configuration's full report is [eval_results.md](eval_results.md). Earlier per-model result files
were removed from the working tree; they remain in the git history (commit `b8f2e83` and earlier).

## Final configuration

| Role | Provider | Model | Reasoning | Free-tier limits that matter |
|---|---|---|---|---|
| Primary | Google AI Studio (OpenAI-compatible endpoint) | `gemini-3.5-flash-lite` | `minimal` | ~250K tokens/min, 15 requests/min (paced locally: no rate-limit headers) |
| Fallback (only one) | Groq | `openai/gpt-oss-120b` | `low` | 8K tokens/min, 200K tokens/day (paced from Groq's rate-limit headers) |

The fallback is used when Gemini returns HTTP 429, a 5xx server error, times out (20 s) or would have to wait
more than 5 s for its rate limit. One switch per turn; the reason is shown in the "Done" line.

**Final eval:** tool choice **7/7**, grounding check 0 failures, 0 untraceable numbers, average turn 7.5 s
(typed questions, 2 LLM calls). Example cards (fast path): Key facts visible in 25-201 ms, full answer
1.3-13 s. Details: [eval_results.md](eval_results.md).

## Why Gemini Flash-Lite as primary (decision 2026-11-18)

- **Rate limits.** On Groq's 8K tokens/minute a single turn (3-8K tokens) could exhaust the budget, so follow-ups
  waited for the window (263 s of pacing wait in one gpt-oss-120b eval run). Gemini Flash-Lite has ~30x more
  per-minute budget: 0 s pacing wait in its evals.
- **Daily quota.** gpt-oss-120b hit Groq's 200K tokens/day limit during development and evaluation. With Gemini as
  primary, Groq's daily budget is kept for the fallback.
- **Model choice.** The Gemini models endpoint was listed once; `gemini-3.5-flash-lite` (version 07-2026) is the
  latest stable Flash-Lite (others: 3.1 stable, 3.1 preview, a "latest" alias). The endpoint does not report
  function-calling support; the evals confirmed it (14/14 correct tool calls across two runs).
- **Thinking level `minimal`** (the lowest) was accepted and gave correct tool choices; Google's guidance to
  keep Gemini 3's default temperature is followed (no temperature is sent).

| Gemini run | Tool choice | Grounding failures | Untraceable numbers | Avg turn | Note |
|---|---|---|---|---|---|
| First run (fallback disabled) | 6/7 graded (7/7 correct tool + arguments) | 0 | 0 | 8.7 s | Turn 5 failed with a Google **HTTP 503** after the correct tool call -> fallback on 5xx/timeouts added |
| Final run (with fallback) | **7/7** | 0 | 0 | 7.5 s | No fallback needed; a 503 during the fast-path measurement was absorbed by Groq |

## Earlier comparison on Groq (before the switch)

| | openai/gpt-oss-120b (low) | qwen/qwen3.8-27b | openai/gpt-oss-20b (after guardrails) |
|---|---|---|---|
| Tool choice | 7/7 (5/5 after the guardrails; rest blocked by the daily limit) | 6/7 (malformed tool call, HTTP 400) | 7/7 |
| Malformed tool calls | 0 | 2 in 2 runs | 0 |
| Avg tokens per turn | 5,124 | 5,639 (max turn > 8K/min) | similar to 120b |
| Untraceable numbers | 0 | 0 | 0 |

`llama-3.3-70b-versatile` was planned but returns *404 model does not exist* for this API key. gpt-oss-120b was
kept as the Groq model; it is now the fallback (gpt-oss-20b is no longer used, so there is exactly one fallback).

## Known LLM limits and mitigations

- **Direction / sign wording.** Models can misstate which way an effect goes (e.g. calling negative growth a
  driver). Mitigations: the **Answer line is computed by code** (rank order, which index is higher, shares,
  scores, capacity gap), a prompt rule ("drives a score UP only if its contribution is positive; never call
  negative growth a driver; use 'raises X' / 'lowers X'"), interpretation strings and precomputed differences in
  the tool outputs. In the final eval the bullets used the sign words correctly; some still repeat signed
  numbers ("lowers LAX ... by -4.22 pts").
- **Weak explanations.** A bullet list can restate the ranking instead of explaining it (final eval, turn 1).
  The facts are correct; the explanation adds little.
- **Rounding in prose** (e.g. "99th percentile" for 99.6) stays within the number check's tolerance.
- **Order and names** can no longer be misquoted in the core answer: both come from the computed Answer line.
- The number check (every number traceable to a tool output) and the grounding check (no invented airports or
  uncalled tools; rankings/comparisons need a tool call, else one retry) run on every LLM-written part.

Limitation: one run per configuration (7 turns each), so differences are indicative, not statistically
significant.
