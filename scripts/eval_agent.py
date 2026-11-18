"""Manual evaluation against the real LLM (Groq free tier): 4 example questions + 3 follow-ups.

Reports per turn: tools called (+ args), whether the tool choice is correct, tokens (provider usage),
pacing waits and number-check flags.  Run:  python scripts/eval_agent.py --effort low|medium|both
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

from agent.agent import Agent, summarize_calls  # noqa: E402
from agent.llm import OpenAIChat  # noqa: E402
from data.repository import load_frames  # noqa: E402
from scoring.config import INVESTMENT_WEIGHTS  # noqa: E402


def _region_ok(args: dict) -> bool:
    return (args.get("region") or "").lower() == "new england" or \
        set(args.get("states") or []) == {"CT", "ME", "MA", "NH", "RI", "VT"}


def _airports(args: dict) -> set[str]:
    return {a.upper() for a in args.get("airports", [])} | ({args["airport"].upper()} if "airport" in args else set())


# (question, acceptable tools, no_tool_ok, argument check on the first acceptable call)
SCRIPT = [
    ("Which airports in New England are strong candidates for terminal expansion?",
     {"rank_airports"}, False, _region_ok),
    ("Rank them again with growth weighted higher.",
     {"rank_airports"}, False,
     lambda a: _region_ok(a) and (a.get("weights") or {}).get("growth", 0) > INVESTMENT_WEIGHTS["growth"]),
    ("Compare LA (LAX) and Santa Ana (SNA) airport congestion levels.",
     {"compare_congestion"}, False, lambda a: {"LAX", "SNA"} <= {x.upper() for x in a.get("airports", [])}
     or {"LA", "SANTA ANA"} <= {x.upper() for x in a.get("airports", [])}),
    ("Why is SNA lower?", {"compare_congestion", "get_airport_profile"}, True, lambda a: True),
    ("What is the percentage of long-haul flights out of Anchorage (ANC)?",
     {"long_haul_share"}, False, lambda a: _airports(a) & {"ANC", "ANCHORAGE"}),
    ("What is the unmet flight demand at SFO and why?",
     {"unmet_demand"}, False, lambda a: _airports(a) & {"SFO", "SAN FRANCISCO"}),
    ("What about Boston?", {"unmet_demand", "get_airport_profile"}, False,
     lambda a: _airports(a) & {"BOS", "BOSTON"}),
]


def grade(result, acceptable: set[str], no_tool_ok: bool, check) -> tuple[bool, str]:
    names = [r.name for r in result.tool_calls]
    if result.error:
        return False, f"error: {result.error}"
    if not names:
        return no_tool_ok, "no tool (answered from context)" if no_tool_ok else "no tool called"
    wrong = [n for n in names if n not in acceptable]
    if wrong:
        return False, f"unexpected tool(s): {wrong}"
    first = next(r for r in result.tool_calls if r.name in acceptable)
    if first.error:
        return False, f"tool error: {first.error}"
    if not check(first.args):
        return False, f"wrong arguments: {first.args}"
    return True, "ok"


def run(effort: str, show_answers: bool) -> dict:
    client = OpenAIChat(reasoning_effort=effort)
    agent = Agent(client, load_frames())
    print(f"\n{'#' * 100}\n# reasoning_effort = {effort}   model = {client.model}\n{'#' * 100}")
    rows = []
    for i, (question, acceptable, no_tool_ok, check) in enumerate(SCRIPT, 1):
        waited_before = client.pacer.total_waited
        t0 = time.monotonic()
        result = agent.ask(question)
        seconds = time.monotonic() - t0
        ok, why = grade(result, acceptable, no_tool_ok, check)
        rows.append({"q": i, "ok": ok, "why": why, "tokens": result.total_tokens, "calls": result.llm_calls,
                     "flags": len(result.untraceable_numbers), "wait": client.pacer.total_waited - waited_before,
                     "seconds": seconds})
        print(f"\n--- [{i}] {question}")
        for line in summarize_calls(result.tool_calls):
            print(f"    tool: {line}")
        print(f"    tool choice: {'CORRECT' if ok else 'WRONG'} ({why}) | {result.llm_calls} LLM calls | "
              f"{result.total_tokens:,} tokens | {seconds:.1f}s (pacing wait {rows[-1]['wait']:.0f}s)")
        if result.untraceable_numbers:
            for n in result.untraceable_numbers:
                print(f"    NUMBER CHECK: untraceable {n['value']!r} in: ...{n['context']}...")
        else:
            print("    number check: all numbers traceable")
        if show_answers:
            print("    answer:\n" + "\n".join("      " + line for line in result.text.splitlines()))
    correct = sum(r["ok"] for r in rows)
    total_tokens = sum(r["tokens"] for r in rows)
    print(f"\nSUMMARY effort={effort}: tool choice {correct}/{len(rows)} correct | total tokens {total_tokens:,} "
          f"(avg {total_tokens // len(rows):,}/turn, max {max(r['tokens'] for r in rows):,}) | "
          f"untraceable numbers {sum(r['flags'] for r in rows)} | pacing wait {sum(r['wait'] for r in rows):.0f}s")
    return {"effort": effort, "correct": correct, "n": len(rows), "tokens": total_tokens,
            "max_turn": max(r["tokens"] for r in rows), "flags": sum(r["flags"] for r in rows),
            "wait": sum(r["wait"] for r in rows), "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--effort", choices=["low", "medium", "both"], default="low")
    parser.add_argument("--no-answers", action="store_true", help="do not print the answer texts")
    args = parser.parse_args()
    load_dotenv()
    efforts = ["low", "medium"] if args.effort == "both" else [args.effort]
    results = []
    for k, effort in enumerate(efforts):
        if k:
            print("\n(waiting 60 s so the token window resets between runs)")
            time.sleep(60)
        results.append(run(effort, not args.no_answers))
    if len(results) > 1:
        print(f"\n{'=' * 100}\nCOMPARISON")
        print(f"{'effort':<8}{'tool choice':>13}{'total tokens':>14}{'avg/turn':>10}{'max turn':>10}"
              f"{'untraceable':>13}{'pacing wait':>13}")
        for r in results:
            print(f"{r['effort']:<8}{r['correct']:>9}/{r['n']:<3}{r['tokens']:>14,}{r['tokens'] // r['n']:>10,}"
                  f"{r['max_turn']:>10,}{r['flags']:>13}{r['wait']:>12.0f}s")
