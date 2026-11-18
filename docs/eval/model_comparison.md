# Model comparison (Groq free tier, 2026-11-18)

Same agent, tools, prompt and question script (`python scripts/eval_agent.py`): the 4 example questions plus
3 follow-ups in one conversation. Full per-turn details and all answers are in:

- [eval_results_gpt-oss-120b.md](eval_results_gpt-oss-120b.md)
- [eval_results_qwen3.8-27b.md](eval_results_qwen3.8-27b.md)

`llama-3.3-70b-versatile` was planned as the comparison model, but Groq returns *404 model does not exist* for
this API key, so `qwen/qwen3.8-27b` (a different model family that supports tool calling) was used instead.
Both models have a 8,000 tokens/minute limit on this key (read from the `x-ratelimit-limit-tokens` header).

## Summary

| | openai/gpt-oss-120b (reasoning_effort=low) | qwen/qwen3.8-27b |
|---|---|---|
| Tool choice correct | **7/7** | 6/7 (turn 2: malformed tool call, HTTP 400) |
| Malformed tool calls across all attempts | 0 | 2 in 2 runs ("rank Airports"; "Failed to call a function") |
| Total tokens (7 turns) | **35,868** | 39,477 |
| Avg / max tokens per turn | **5,124** / 7,585 | 5,639 / 9,916 (max turn > 8K/min limit) |
| Rate-limit pacing wait | 263 s | 238 s |
| Untraceable numbers | 0 | 0 |
| Q4 SFO answer | Correct, concise; omits the capacity gap | Correct and most complete: capacity gap correctly labelled per target, component meanings used |
| "Why is SNA lower?" | **Direction errors**: says SNA's higher NAS share "reduces" its index and that SNA has "slightly higher" ops/runway (both wrong) | Opening sentence wrong ("SNA is not lower ... slightly higher"), but the driver bullets are correct, incl. which index each component raises |

## Recommendation

**Keep `openai/gpt-oss-120b` (reasoning_effort=low) as the default.**

- In an agent, tool-call reliability is the critical property: a malformed call means the user gets an error
  instead of an answer. gpt-oss made 0 malformed calls; Qwen made 2 in 2 runs.
- gpt-oss uses ~9% fewer tokens, and its largest turn stays under the 8K tokens/minute limit.
- Qwen writes better explanations (Q4 especially), but both models still make reasoning mistakes about the
  direction of effects. This is mitigated the same way for both: every number is computed by the tools and
  traceable (0 untraceable numbers for both), and the UI shows the raw tool tables next to each answer.
- `qwen/qwen3.8-27b` remains available via `LLM_MODEL=qwen/qwen3.8-27b` (no code change).

## Known LLM reasoning limits (to document in DESIGN.md)

- Models may misstate the **direction** of an effect even when the tool output states it explicitly
  ("raises SNA's index"). Numbers stay correct; interpretations can be wrong.
- Models may add **causal links** between independent score components.
- Mitigations in place: deterministic interpretation strings, precomputed differences and ranked drivers,
  component meanings, units on every value, the number-traceability check, and raw tool tables in the UI.

Limitation of this comparison: one run per model (7 turns each), so differences are indicative, not statistically
significant.

## Update: after the grounding guardrails (2026-11-18, later run)

A manual test found an invented re-ranking ("Rank them again with growth weighted higher" answered without a
tool call). Fix: structured follow-up context (previous tool call + args injected into each turn), a
deterministic grounding check (airport codes grounded, no claimed-but-uncalled tools, rank/compare/weight
requests need a tool call) and one retry with `tool_choice="required"`.

| | openai/gpt-oss-120b | openai/gpt-oss-20b |
|---|---|---|
| Tool choice | 5/5 completed turns correct; turns 6-7 blocked by the provider's **daily** token limit | **7/7** |
| Grounding check | passed on all completed turns, 0 retries | **passed on all 7 turns, 0 retries** |
| Untraceable numbers | 0 | 0 |
| Follow-up "Rank them again with growth weighted higher" | correct tool call (`region=new england`, growth 0.5); answer text skipped HVN/BTV in the order (wrong #3) | correct tool call and correct order (BOS, PWM, HVN); wrong airport name for HVN |

- gpt-oss-120b log: [eval_log_gpt-oss-120b_after_guardrails.txt](eval_log_gpt-oss-120b_after_guardrails.txt)
  (the full results file above is from the earlier run; a complete re-run was blocked by the daily limit).
- gpt-oss-20b full results: [eval_results_gpt-oss-20b.md](eval_results_gpt-oss-20b.md).

**Decision:** gpt-oss-20b met the criteria (7/7, grounding passed on every turn) and is enabled as the fallback
(`LLM_FALLBACK_MODEL=openai/gpt-oss-20b`). Groq's free tier allows **200,000 tokens per day per model**; the
fallback has its own daily budget, so it also roughly doubles daily capacity.

Remaining limit (documented, not fixed by prompt tuning): both models can still misstate an **order** or an
airport **name** while every number and code is correct. The checks do not catch this; the Key facts box under
each answer shows the deterministic ranking.
