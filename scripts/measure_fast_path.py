"""Measure the fast path of the example cards against the real LLM (no browser).

For each example card, the same steps as the app (app.answer_direct), in a fresh conversation, without the answer
cache:
  click -> tool runs -> Answer headline + Key facts + Evidence tables ready   ("Key facts visible")
        -> one LLM call writes the Why / drivers bullets                       ("full answer")
Streamlit's own rendering time (tens of milliseconds) is not included.
Appends a "Fast path" section to docs/eval/eval_results.md (replacing an earlier one).
Run:  python scripts/measure_fast_path.py [--pause 60]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

from agent.agent import Agent  # noqa: E402
from agent.headlines import answer_headline  # noqa: E402
from agent.llm import OpenAIChat  # noqa: E402
from data.repository import load_frames  # noqa: E402
from ui import evidence  # noqa: E402
from ui.key_facts import key_facts  # noqa: E402
from ui.render import EXAMPLE_CALLS, EXAMPLES  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "docs" / "eval" / "eval_results.md"
MARKER = "## Fast path (example cards)"


def measure() -> list[dict]:
    frames = load_frames()
    client = OpenAIChat.from_env()
    rows = []
    for question, (tool, args) in zip(EXAMPLES, EXAMPLE_CALLS):
        agent = Agent(client, frames)                       # fresh conversation, as after a click on a card
        t0 = time.perf_counter()
        turn = agent.start_direct(question, tool, args)
        answer_headline(turn.records)
        key_facts(turn.records)
        for record in turn.records:
            evidence.tables(record)
            evidence.chart(record)
        t_facts = time.perf_counter() - t0
        result = agent.explain(turn)
        t_full = time.perf_counter() - t0
        rows.append({"question": question, "tool": tool, "facts": t_facts, "full": t_full,
                     "models": result.models, "fallback": result.fallback_reason, "error": result.error,
                     "tokens": result.total_tokens, "flags": len(result.untraceable_numbers),
                     "grounding": result.grounding_issues})
        print(f"{tool:<20} key facts {t_facts * 1000:7.0f} ms | full answer {t_full:5.1f} s | "
              f"{' + '.join(result.models)}" + (f" (fallback: {result.fallback_reason})" if result.fallback_reason else "")
              + (f" | ERROR {result.error}" if result.error else ""))
    return rows


def write(rows: list[dict]) -> None:
    lines = [MARKER, "",
             "Measured by `python scripts/measure_fast_path.py`: an example card runs its known tool directly, shows "
             "the computed Answer, Key facts and Evidence, then one LLM call writes the bullets. Fresh conversation "
             "per card, answer cache not used; Streamlit rendering time not included.", "",
             "| example | tool | Key facts visible | full answer | model | tokens | checks |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        model = " + ".join(m.split("/")[-1] for m in r["models"]) + (f" (fallback: {r['fallback']})" if r["fallback"] else "")
        checks = "error: " + r["error"] if r["error"] else (
            "grounding FAILED" if r["grounding"] else f"passed, {r['flags']} untraceable numbers")
        lines.append(f"| {r['question']} | `{r['tool']}` | {r['facts'] * 1000:.0f} ms | {r['full']:.1f} s | {model} | "
                     f"{r['tokens']:,} | {checks} |")
    full = [r["full"] for r in rows]
    facts = [r["facts"] * 1000 for r in rows]
    lines += ["", f"Key facts visible: avg {sum(facts) / len(facts):.0f} ms, max {max(facts):.0f} ms. "
                  f"Full answer: avg {sum(full) / len(full):.1f} s, max {max(full):.1f} s."]
    text = RESULTS.read_text(encoding="utf-8") if RESULTS.exists() else "# Agent evaluation results\n"
    text = text.split(MARKER)[0].rstrip() + "\n\n" + "\n".join(lines) + "\n"
    RESULTS.write_text(text, encoding="utf-8")
    print(f"\nSaved {RESULTS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pause", type=float, default=0, help="seconds to wait first (e.g. after the eval)")
    args = parser.parse_args()
    load_dotenv()
    if args.pause:
        time.sleep(args.pause)
    write(measure())
