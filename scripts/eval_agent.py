"""Manual evaluation against the real LLM (Groq free tier): 4 example questions + 3 follow-ups.

Reports per turn: tools called (+ args), whether the tool choice is correct, tokens (provider usage),
pacing waits and number-check flags, and writes everything (incl. final answers) to docs/eval/eval_results.md.
Run:  python scripts/eval_agent.py --effort low|medium|both
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

from agent.agent import Agent, summarize_calls  # noqa: E402
from agent.llm import OpenAIChat  # noqa: E402
from data.repository import load_frames  # noqa: E402
from scoring.config import INVESTMENT_WEIGHTS  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parents[1] / "docs" / "eval"


def results_path(model: str) -> Path:
    """One results file per model, e.g. docs/eval/eval_results_gpt-oss-120b.md."""
    return EVAL_DIR / f"eval_results_{model.split('/')[-1]}.md"


def _region_ok(args: dict) -> bool:
    return (args.get("region") or "").lower() == "new england" or \
        set(args.get("states") or []) == {"CT", "ME", "MA", "NH", "RI", "VT"}


def _rerank_ok(args: dict) -> bool:
    """Expected for "Rank them again with growth weighted higher": the SAME region (New England, as in the
    previous turn) and a growth weight above the default."""
    return _region_ok(args) and (args.get("weights") or {}).get("growth", 0) > INVESTMENT_WEIGHTS["growth"]


def _airports(args: dict) -> set[str]:
    return {a.upper() for a in args.get("airports", [])} | ({args["airport"].upper()} if "airport" in args else set())


# (question, acceptable tools, no_tool_ok, argument check on the first acceptable call)
SCRIPT = [
    ("Which airports in New England are strong candidates for terminal expansion?",
     {"rank_airports"}, False, _region_ok),
    ("Rank them again with growth weighted higher.",
     {"rank_airports"}, False, _rerank_ok),   # expected: region = new england, growth weight > default
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
    if result.grounding_issues:
        return False, f"grounding check failed: {'; '.join(result.grounding_issues)}"
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
    client = OpenAIChat(reasoning_effort=effort, fallback_model="")   # measure one model alone
    agent = Agent(client, load_frames())
    print(f"\n{'#' * 100}\n# reasoning_effort = {effort}   model = {client.model}\n{'#' * 100}")
    rows = []
    for i, (question, acceptable, no_tool_ok, check) in enumerate(SCRIPT, 1):
        waited_before = client.pacer.total_waited
        t0 = time.monotonic()
        result = agent.ask(question)
        if result.error == "rate_limited":          # eval only: give the window time to reset, retry once
            print(f"    (rate-limited on [{i}]; waiting 60 s and retrying once)")
            time.sleep(60)
            result = agent.ask(question)
        seconds = time.monotonic() - t0
        ok, why = grade(result, acceptable, no_tool_ok, check)
        rows.append({"q": i, "question": question, "ok": ok, "why": why, "tokens": result.total_tokens,
                     "calls": result.llm_calls, "flags": len(result.untraceable_numbers),
                     "untraceable": result.untraceable_numbers, "wait": client.pacer.total_waited - waited_before,
                     "seconds": seconds, "tools": [(r.name, r.args, r.error) for r in result.tool_calls],
                     "answer": result.text, "grounding": result.grounding_issues,
                     "retried": result.retried, "models": result.models})
        print(f"\n--- [{i}] {question}")
        for line in summarize_calls(result.tool_calls):
            print(f"    tool: {line}")
        print(f"    tool choice: {'CORRECT' if ok else 'WRONG'} ({why}) | {result.llm_calls} LLM calls | "
              f"{result.total_tokens:,} tokens | {seconds:.1f}s (pacing wait {rows[-1]['wait']:.0f}s)")
        print(f"    grounding check: {'FAILED ' + '; '.join(result.grounding_issues) if result.grounding_issues else 'passed'}"
              + (" (after an automatic retry)" if result.retried else ""))
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
          f"untraceable numbers {sum(r['flags'] for r in rows)} | grounding failures "
          f"{sum(bool(r['grounding']) for r in rows)} | retries {sum(r['retried'] for r in rows)} | "
          f"pacing wait {sum(r['wait'] for r in rows):.0f}s")
    return {"effort": effort, "model": client.model, "max_output_tokens": client.max_output_tokens,
            "tpm_limit": client.pacer.limit, "reasoning_effort_sent": client.reasoning_effort,
            "correct": correct, "n": len(rows), "tokens": total_tokens,
            "max_turn": max(r["tokens"] for r in rows), "flags": sum(r["flags"] for r in rows),
            "grounding_failures": sum(bool(r["grounding"]) for r in rows), "retries": sum(r["retried"] for r in rows),
            "wait": sum(r["wait"] for r in rows), "rows": rows}


def write_markdown(results: list[dict], path: Path | None = None) -> None:
    """Save the eval (summary, per-turn details and final answers) as markdown in the repo."""
    path = path or results_path(results[0]["model"])
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Agent evaluation results",
        "",
        f"Generated by `python scripts/eval_agent.py` on {datetime.now():%Y-%m-%d %H:%M}. "
        f"Model `{results[0]['model']}`, max output tokens per call {results[0]['max_output_tokens']}, "
        f"provider token limit {results[0]['tpm_limit']:,}/min. "
        "One conversation per run: the 4 example questions plus 3 follow-ups.",
        "",
        "Tokens = provider-reported usage (incl. hidden reasoning tokens). Pacing wait = time the client waited "
        "to stay under the free tier's tokens-per-minute limit.",
        "",
        "## Summary",
        "",
        "| reasoning effort (sent) | tool choice correct | grounding failures | retries | total tokens | avg / turn "
        "| max turn | untraceable numbers | pacing wait |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['reasoning_effort_sent'] or 'n/a (not supported)'} | {r['correct']}/{r['n']} | "
                     f"{r['grounding_failures']} | {r['retries']} | {r['tokens']:,} | {r['tokens'] // r['n']:,} | "
                     f"{r['max_turn']:,} | {r['flags']} | {r['wait']:.0f} s |")
    for r in results:
        lines += ["", f"## Run: reasoning_effort = {r['reasoning_effort_sent'] or 'n/a'}", "",
                  "| # | question | tools called | tool choice | grounding | LLM calls | tokens | wait | number check |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for row in r["rows"]:
            tools = "<br>".join(f"`{n}({json.dumps(a, ensure_ascii=False)})`" + (f" ERROR {e}" if e else "")
                                for n, a, e in row["tools"]) or "none"
            check = "all traceable" if not row["untraceable"] else ", ".join(u["value"] for u in row["untraceable"])
            grounding = ("FAILED: " + "; ".join(row["grounding"])) if row["grounding"] else "passed"
            grounding += " (retried)" if row["retried"] else ""
            lines.append(f"| {row['q']} | {row['question']} | {tools} | {'correct' if row['ok'] else 'WRONG'} "
                         f"({row['why']}) | {grounding} | {row['calls']} | {row['tokens']:,} | {row['wait']:.0f} s | {check} |")
        lines += ["", "### Answers"]
        for row in r["rows"]:
            lines += ["", f"#### {row['q']}. {row['question']}", ""]
            lines += [f"> {line}" if line.strip() else ">" for line in row["answer"].splitlines()]
            for u in row["untraceable"]:
                lines += ["", f"_Untraceable number_ `{u['value']}`: ...{u['context']}..."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved {path}")


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
    write_markdown(results)
    if len(results) > 1:
        print(f"\n{'=' * 100}\nCOMPARISON")
        print(f"{'effort':<8}{'tool choice':>13}{'total tokens':>14}{'avg/turn':>10}{'max turn':>10}"
              f"{'untraceable':>13}{'pacing wait':>13}")
        for r in results:
            print(f"{r['effort']:<8}{r['correct']:>9}/{r['n']:<3}{r['tokens']:>14,}{r['tokens'] // r['n']:>10,}"
                  f"{r['max_turn']:>10,}{r['flags']:>13}{r['wait']:>12.0f}s")
