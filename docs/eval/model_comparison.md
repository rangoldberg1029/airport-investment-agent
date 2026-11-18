# Language models: choice and evaluation

All runs use the same agent, tools, deterministic scoring and question script (`python scripts/eval_agent.py`):
the 4 example questions plus 3 follow-ups in one conversation. Only free tiers are used. The full report of the
evaluated configuration is [eval_results.md](eval_results.md).

## Configuration

| Role | Provider | Model (env var, default) | Reasoning | Free-tier limits that matter |
|---|---|---|---|---|
| Language model (required) | Google AI Studio, OpenAI-compatible endpoint | `GEMINI_MODEL`, `gemini-3.5-flash-lite` | `minimal` | ~250K tokens/min, 15 requests/min (paced locally; no rate-limit headers) |
| Failover (optional) | Groq | `GROQ_MODEL`, `openai/gpt-oss-120b` | `low` | 8K tokens/min, 200K tokens/day (paced from Groq's rate-limit headers) |

Keys: `GEMINI_API_KEY` (required), `GROQ_API_KEY` (optional; also enables voice input via Whisper on Groq).
Base URLs, reasoning levels, limits and timeouts are constants in `agent/llm.py`.

| Keys set | Behaviour |
|---|---|
| Both | Gemini answers; Groq takes over per the error policy below; voice input on |
| `GEMINI_API_KEY` only | Gemini answers; no failover; voice input off ("Voice input requires GROQ_API_KEY") |
| `GROQ_API_KEY` only | Groq answers alone; voice input on |
| Neither | Setup card; data, methodology and About panel still available |

## Error policy (per LLM call)

| Gemini result | Action |
|---|---|
| HTTP 429 (per-minute or daily limit) | Groq immediately |
| HTTP 5xx, timeout (20 s) or connection error | Retry Gemini once after 2 s; if it fails again, Groq |
| Would wait > 5 s for its own rate limit | Groq |
| Other errors (e.g. HTTP 400) | Shown as an error (Groq would fail the same way) |

Once a turn uses Groq it stays on Groq for the rest of that turn; the next turn starts on Gemini again. The
"Done" line under each answer names the model that answered and, after a failover, why. A model whose daily
quota is used up is skipped for the rest of the session. Without Groq, a 429 is retried with backoff (at most
30 s) and a 5xx/timeout is retried once; after that the turn shows an error.

## Why these models

- **Per-minute budget.** One turn uses 3-8K tokens. Gemini Flash-Lite's ~250K tokens/min means no pacing waits
  (0 s in the evals); Groq's 8K tokens/min would make follow-ups wait for the window.
- **Daily quota.** Groq allows 200K tokens/day per model; with Gemini answering, Groq's budget is kept for
  failover and voice.
- **Model version.** `gemini-3.5-flash-lite` (07-2026) is the latest stable Flash-Lite in Google's models list
  (also listed: 3.1 stable, 3.1 preview, a "latest" alias). The list does not report function-calling support;
  the evals confirm it (14/14 correct tool calls in two 7-turn runs).
- **Thinking level `minimal`**, the lowest, gives correct tool choices. No temperature is sent to Gemini 3
  (Google advises keeping its default).
- **Server errors happen.** Gemini returned HTTP 503 during evaluation runs, which is why a 5xx is retried once
  and then handed to Groq.
- **Groq model.** `openai/gpt-oss-120b` made 0 malformed tool calls in its evals (7/7 tool choice).

## Models evaluated

| Model | Tool choice | Malformed tool calls | Untraceable numbers | Used as |
|---|---|---|---|---|
| `gemini-3.5-flash-lite` (minimal) | 7/7 | 0 | 0 | language model |
| `openai/gpt-oss-120b` (low) | 7/7 | 0 | 0 | failover |
| `qwen/qwen3.8-27b` | 6/7 | 2 in 2 runs (HTTP 400) | 0 | not used |
| `llama-3.3-70b-versatile` | - | - | - | not available on the evaluation key (HTTP 404) |

## Known LLM limits and mitigations

- **The core answer is computed by code.** The *Answer* line (rank order, names, which index is higher,
  shares, scores, capacity gap, weight preset) comes from the tool output, so it cannot be misquoted.
- **Re-rankings are deterministic.** Requests like "weight growth higher" map to a named weight preset
  (`growth-led`); custom weights are used only when the user gives numbers.
- **Direction / sign wording in the bullets.** A model can still word an effect loosely. Mitigations: the
  prompt rule "drives a score UP only if its contribution is positive; never call negative growth a driver;
  use 'raises X' / 'lowers X'", interpretation strings and precomputed differences in the tool outputs.
- **Weak explanations.** Bullets can restate the ranking instead of explaining it; the facts stay correct.
- **Checks on every LLM-written part:** every number must trace to a tool output; no invented airports or
  uncalled tools; rankings, comparisons and weight changes need a tool call (else one retry).

Limitation: one run per configuration (7 turns each), so differences are indicative, not statistically
significant.
