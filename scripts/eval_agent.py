"""Manual evaluation against the real LLMs (the app's configuration: Gemini + Groq failover): 4 example questions
+ 3 follow-ups + 3 out-of-scope questions, one conversation.

Reports per turn: tools called (+ args), whether the turn passes (correct tool choice; for out-of-scope questions:
a clear refusal, no invented numbers and an in-scope offer naming the question's US airport), grounding and number checks,
tokens (provider usage), latency, pacing waits and which model answered (with the failover reason, if any).
Writes everything (incl. final answers) to docs/eval/eval_results.md.
Run:  python scripts/eval_agent.py [--no-fallback] [--effort minimal|low|medium]
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

from agent.agent import Agent, summarize_calls  # noqa: E402
from agent.guardrails import is_refusal  # noqa: E402
from agent.llm import GEMINI_MAX_OUTPUT_TOKENS, OpenAIChat  # noqa: E402
from data.repository import load_frames  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parents[1] / "docs" / "eval"


EVAL_RESULTS = EVAL_DIR / "eval_results.md"


def _secs(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f} s"


def _region_ok(args: dict) -> bool:
    return (args.get("region") or "").lower() == "new england" or \
        set(args.get("states") or []) == {"CT", "ME", "MA", "NH", "RI", "VT"}


def _rerank_ok(args: dict) -> bool:
    """Expected for "Rank them again with growth weighted higher": the SAME region (New England, as in the
    previous turn) and the named preset growth-led (no explicit numbers were given, so no custom weights)."""
    return _region_ok(args) and args.get("preset") == "growth-led" and not args.get("weights")


REFUSAL = "refusal"     # marker: an out-of-scope question, graded as a refusal (see grade_refusal)


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
    # Out-of-scope questions (acceptable tools = REFUSAL): graded by grade_refusal; the last item lists the US
    # airports the in-scope offer must name.
    ("What would it cost to build a new terminal at BOS?", REFUSAL, True, ("BOS", "Boston")),
    ("Compare London Heathrow and JFK congestion.", REFUSAL, True, ("JFK",)),
    ("How many passengers will LAX handle in 2030?", REFUSAL, True, ("LAX", "Los Angeles")),
]

OFFER_RE = re.compile(r"\b(I can|I could|can show|could show|happy to|is in scope|are in scope)\b", re.I)


def grade_refusal(result, entities: tuple[str, ...]) -> tuple[bool, str]:
    """Pass = a clear refusal (scope statement), no invented numbers, and an in-scope offer naming the
    question's US airport."""
    text = result.text or ""
    if result.error:
        return False, f"error: {result.error}"
    if result.grounding_issues:
        return False, f"grounding check failed: {'; '.join(result.grounding_issues)}"
    if not (is_refusal(text) or any(r.meta.get("out_of_scope") for r in result.tool_calls)):
        return False, "no clear refusal"
    if result.untraceable_numbers:
        return False, "invented numbers: " + ", ".join(n["value"] for n in result.untraceable_numbers)
    if not OFFER_RE.search(text) or not any(e.lower() in text.lower() for e in entities):
        return False, "no in-scope offer naming " + "/".join(entities)
    sentences = len([s for s in re.split(r"(?<=[.!?])\s+", result.explanation.strip()) if s])
    return True, f"refusal + in-scope offer ({sentences} sentence{'s' if sentences != 1 else ''})"


def grade(result, acceptable, no_tool_ok: bool, check) -> tuple[bool, str]:
    if acceptable is REFUSAL:
        return grade_refusal(result, check)
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


def run(effort: str | None, show_answers: bool, use_fallback: bool = True) -> dict:
    client = OpenAIChat.from_env(use_fallback=use_fallback)
    if effort:
        client.primary.effort = effort
    effort = client.reasoning_effort or "n/a"
    agent = Agent(client, load_frames())
    print(f"\n{'#' * 100}\n# reasoning_effort = {effort}   model = {client.model}   "
          f"failover = {client.fallback_model or 'none'}\n{'#' * 100}")
    rows = []
    for i, (question, acceptable, no_tool_ok, check) in enumerate(SCRIPT, 1):
        waited_before = client.pacer.total_waited
        marks: dict[str, float] = {}

        def on_event(event: str, **kw) -> None:
            """Time to Key facts (tools ran) and to the first streamed bullet token."""
            if event == "facts_ready":
                marks.setdefault("facts", time.monotonic() - t0)
            elif event == "token":
                marks.setdefault("first", time.monotonic() - t0)
        t0 = time.monotonic()
        result = agent.ask(question, on_event=on_event)
        if result.error == "rate_limited":          # eval only: give the window time to reset, retry once
            print(f"    (rate-limited on [{i}]; waiting 60 s and retrying once)")
            time.sleep(60)
            marks.clear()
            t0 = time.monotonic()
            result = agent.ask(question, on_event=on_event)
        seconds = time.monotonic() - t0
        ok, why = grade(result, acceptable, no_tool_ok, check)
        rows.append({"q": i, "question": question, "ok": ok, "why": why, "tokens": result.total_tokens,
                     "calls": result.llm_calls, "flags": len(result.untraceable_numbers),
                     "untraceable": result.untraceable_numbers, "wait": client.pacer.total_waited - waited_before,
                     "seconds": seconds, "tools": [(r.name, r.args, r.error) for r in result.tool_calls],
                     "answer": result.text, "grounding": result.grounding_issues,
                     "retried": result.retried, "models": result.models,
                     "fallback_reason": result.fallback_reason,
                     "facts_s": marks.get("facts"), "first_s": marks.get("first")})
        print(f"\n--- [{i}] {question}")
        for line in summarize_calls(result.tool_calls):
            print(f"    tool: {line}")
        print(f"    model: {' + '.join(result.models) or 'none'}"
              + (f" (failover: {result.fallback_reason})" if result.fallback_reason else ""))
        print(f"    result: {'PASS' if ok else 'FAIL'} ({why}) | {result.llm_calls} LLM calls | "
              f"{result.total_tokens:,} tokens | Key facts {_secs(rows[-1]['facts_s'])}, first bullet "
              f"{_secs(rows[-1]['first_s'])}, full {seconds:.1f}s (pacing wait {rows[-1]['wait']:.0f}s)")
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
    print(f"\nSUMMARY effort={effort}: {correct}/{len(rows)} pass | total tokens {total_tokens:,} "
          f"(avg {total_tokens // len(rows):,}/turn, max {max(r['tokens'] for r in rows):,}) | "
          f"untraceable numbers {sum(r['flags'] for r in rows)} | grounding failures "
          f"{sum(bool(r['grounding']) for r in rows)} | retries {sum(r['retried'] for r in rows)} | "
          f"pacing wait {sum(r['wait'] for r in rows):.0f}s | latency avg "
          f"{sum(r['seconds'] for r in rows) / len(rows):.1f}s, max {max(r['seconds'] for r in rows):.1f}s")
    return {"effort": effort, "model": client.model, "fallback_model": client.fallback_model,
            "max_output_tokens": (max(client.max_output_tokens, GEMINI_MAX_OUTPUT_TOKENS) if client.primary.is_gemini
                                  else client.max_output_tokens),
            "fallbacks": sum(bool(r["fallback_reason"]) for r in rows),
            "latency_avg": sum(r["seconds"] for r in rows) / len(rows), "latency_max": max(r["seconds"] for r in rows),
            "tpm_limit": client.pacer.limit, "reasoning_effort_sent": client.reasoning_effort,
            "correct": correct, "n": len(rows), "tokens": total_tokens,
            "max_turn": max(r["tokens"] for r in rows), "flags": sum(r["flags"] for r in rows),
            "grounding_failures": sum(bool(r["grounding"]) for r in rows), "retries": sum(r["retried"] for r in rows),
            "wait": sum(r["wait"] for r in rows), "rows": rows}


def write_markdown(results: list[dict], path: Path | None = None) -> None:
    """Save the eval (summary, per-turn details and final answers) as markdown in the repo."""
    path = path or EVAL_RESULTS
    path.parent.mkdir(parents=True, exist_ok=True)
    first = results[0]
    lines = [
        "# Agent evaluation results",
        "",
        f"Generated by `python scripts/eval_agent.py` on {datetime.now():%Y-%m-%d %H:%M}. Primary model "
        f"`{first['model']}` (reasoning effort `{first['reasoning_effort_sent'] or 'n/a'}`), failover "
        f"`{first['fallback_model'] or 'none'}`; max output tokens per primary call {first['max_output_tokens']:,}. "
        "One conversation: the 4 example questions, 3 follow-ups and 3 out-of-scope questions, typed (the LLM "
        "chooses the tools). A data question passes with the correct tool and arguments; an out-of-scope "
        "question passes with a clear refusal, no invented numbers and an in-scope offer naming the question's "
        "US airport.",
        "",
        "Every answer starts with a computed **Answer** line (built by code from the tool output); the LLM writes "
        "only the *Why / drivers* bullets, streamed. The grounding and number checks run on the LLM part. Tokens = "
        "provider-reported usage (incl. hidden reasoning tokens; earlier turns' tool outputs are not resent, only "
        "a one-line summary). Key facts = time until the tool ran and the Answer line, Key facts and Evidence "
        "are shown; first bullet = time to the first streamed token; full = wall time of the whole turn. Pacing "
        "wait = time the client waited to stay under the primary's rate limits.",
        "",
        "## Summary",
        "",
        "| pass | grounding failures | retries | untraceable numbers | failover turns "
        "| latency avg / max | total tokens | avg / turn | pacing wait |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['correct']}/{r['n']} | {r['grounding_failures']} | {r['retries']} | {r['flags']} | "
                     f"{r['fallbacks']} | {r['latency_avg']:.1f} s / {r['latency_max']:.1f} s | {r['tokens']:,} | "
                     f"{r['tokens'] // r['n']:,} | {r['wait']:.0f} s |")
    for r in results:
        lines += ["", "## Per turn", "",
                  "| # | question | tools called | result | grounding | model | LLM calls | tokens | Key facts "
                  "| first bullet | full | wait | number check |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for row in r["rows"]:
            tools = "<br>".join(f"`{n}({json.dumps(a, ensure_ascii=False)})`" + (f" ERROR {e}" if e else "")
                                for n, a, e in row["tools"]) or "none"
            check = "all traceable" if not row["untraceable"] else ", ".join(u["value"] for u in row["untraceable"])
            grounding = ("FAILED: " + "; ".join(row["grounding"])) if row["grounding"] else "passed"
            grounding += " (retried)" if row["retried"] else ""
            model = " + ".join(m.split("/")[-1] for m in row["models"]) or "none"
            if row["fallback_reason"]:
                model += f" (failover: {row['fallback_reason']})"
            lines.append(f"| {row['q']} | {row['question']} | {tools} | {'pass' if row['ok'] else 'FAIL'} "
                         f"({row['why']}) | {grounding} | {model} | {row['calls']} | {row['tokens']:,} | "
                         f"{_secs(row['facts_s'])} | {_secs(row['first_s'])} | {row['seconds']:.1f} s | "
                         f"{row['wait']:.0f} s | {check} |")
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
    parser.add_argument("--effort", choices=["minimal", "low", "medium", "both"], default=None,
                        help="reasoning effort of the primary model; default: the configured constant")
    parser.add_argument("--no-answers", action="store_true", help="do not print the answer texts")
    parser.add_argument("--no-fallback", action="store_true", help="measure the primary model alone (no failover)")
    args = parser.parse_args()
    load_dotenv()
    efforts = ["low", "medium"] if args.effort == "both" else [args.effort]
    results = []
    for k, effort in enumerate(efforts):
        if k:
            print("\n(waiting 60 s so the token window resets between runs)")
            time.sleep(60)
        results.append(run(effort, not args.no_answers, use_fallback=not args.no_fallback))
    write_markdown(results)
    if len(results) > 1:
        print(f"\n{'=' * 100}\nCOMPARISON")
        print(f"{'effort':<8}{'tool choice':>13}{'total tokens':>14}{'avg/turn':>10}{'max turn':>10}"
              f"{'untraceable':>13}{'pacing wait':>13}")
        for r in results:
            print(f"{r['effort']:<8}{r['correct']:>9}/{r['n']:<3}{r['tokens']:>14,}{r['tokens'] // r['n']:>10,}"
                  f"{r['max_turn']:>10,}{r['flags']:>13}{r['wait']:>12.0f}s")
