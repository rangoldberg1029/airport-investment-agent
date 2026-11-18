"""Print the token size of everything the LLM sees: system prompt, tool schemas, tool outputs per question.

Estimate = characters / 4 (no tokenizer dependency). Exact counts come from the provider's `usage`
(see scripts/eval_agent.py). Run:  python scripts/token_budget.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.prompts import SYSTEM_PROMPT  # noqa: E402
from agent.tools import TOOL_SCHEMAS, execute_tool  # noqa: E402
from data.repository import load_frames  # noqa: E402

QUESTIONS = {
    "Q1 New England candidates": [("rank_airports", {"region": "new england"})],
    "Q2 LAX vs SNA congestion": [("compare_congestion", {"airports": ["LA", "Santa Ana"]})],
    "Q3 ANC long-haul share": [("long_haul_share", {"airport": "Anchorage"})],
    "Q4 SFO unmet demand": [("unmet_demand", {"airport": "SFO"})],
    "Follow-up: What about Boston?": [("get_airport_profile", {"airport": "Boston"})],
    "Methodology: data sources": [("explain_methodology", {"topic": "data_sources"})],
    "Live status SFO": [("live_airport_status", {"airport": "SFO"})],
}
ANSWER_ESTIMATE = 450      # typical final answer (tokens)
TPM = 8000


def tok(text: str) -> int:
    return len(text) // 4


if __name__ == "__main__":
    frames = load_frames()
    prompt = tok(SYSTEM_PROMPT)
    schemas = tok(json.dumps(TOOL_SCHEMAS, separators=(",", ":")))
    fixed = prompt + schemas
    print(f"System prompt:        ~{prompt:>5} tokens")
    print(f"Tool schemas (7):     ~{schemas:>5} tokens")
    print(f"Fixed per LLM call:   ~{fixed:>5} tokens\n")
    print(f"{'question':<32}{'tool output':>12}{'turn total*':>13}{'% of 8K TPM':>13}")
    for label, calls in QUESTIONS.items():
        out = sum(tok(execute_tool(n, a, frames).llm_view) for n, a in calls)
        # call 1: fixed + question(~30) -> tool call (~60); call 2: fixed + question + tool call + output -> answer
        turn = (fixed + 30 + 60) + (fixed + 30 + 60 + out + ANSWER_ESTIMATE)
        print(f"{label:<32}{out:>12,}{turn:>13,}{turn / TPM:>12.0%}")
    print(f"\n* estimate for one turn = 2 LLM calls (choose tool, then answer ~{ANSWER_ESTIMATE} tokens), "
          "excluding hidden reasoning tokens; exact usage is reported by scripts/eval_agent.py")
