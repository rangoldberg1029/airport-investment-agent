"""Measure response times of the 4 example questions against the real LLM (no browser), two ways:

  card click (fast path): the known tool runs -> Answer line, Key facts and Evidence ready ("Key facts")
                          -> the streamed explanation starts ("first bullet") -> ends ("full answer")
  typed question:         LLM chooses the tool -> tool runs -> "Key facts" -> streamed explanation as above

Same steps as the app, fresh conversation per question, answer cache not used. Streamlit's own rendering time
(tens of milliseconds) is not included. Appends a "Response times" section to docs/eval/eval_results.md
(replacing an earlier one).
Run:  python scripts/measure_fast_path.py [--pause 60]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

from agent.agent import Agent  # noqa: E402
from agent.llm import OpenAIChat  # noqa: E402
from data.repository import load_frames  # noqa: E402
from ui import evidence  # noqa: E402
from ui.key_facts import key_facts  # noqa: E402
from ui.render import EXAMPLE_CALLS, EXAMPLES  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "docs" / "eval" / "eval_results.md"
MARKER = "## Response times (example questions)"


def _prepare_display(records) -> None:
    """What the app computes before showing Key facts and Evidence."""
    key_facts(records)
    for record in records:
        evidence.tables(record)
        evidence.chart(record)


def measure_one(client, frames, question: str, tool: str | None, args: dict) -> dict:
    agent = Agent(client, frames)                         # fresh conversation
    marks: dict[str, float] = {}
    t0 = time.perf_counter()

    def on_event(event: str, **kw) -> None:
        if event == "facts_ready":
            _prepare_display(kw["records"])
            marks.setdefault("facts", time.perf_counter() - t0)
        elif event == "token":
            marks.setdefault("first", time.perf_counter() - t0)
    if tool:
        turn = agent.start_direct(question, tool, args)
        agent.headline(turn)
        _prepare_display(turn.records)
        marks["facts"] = time.perf_counter() - t0
        result = agent.explain(turn, on_event=on_event)
    else:
        result = agent.ask(question, on_event=on_event)
    return {"question": question, "path": "card click" if tool else "typed", "facts": marks.get("facts"),
            "first": marks.get("first"), "full": time.perf_counter() - t0, "models": result.models,
            "fallback": result.fallback_reason, "error": result.error, "tokens": result.total_tokens,
            "flags": len(result.untraceable_numbers), "grounding": result.grounding_issues,
            "retried": result.retried}


def measure() -> list[dict]:
    frames = load_frames()
    client = OpenAIChat.from_env()
    rows = []
    for question, (tool, args) in zip(EXAMPLES, EXAMPLE_CALLS):
        for path_tool in (tool, None):                     # card click, then typed
            row = measure_one(client, frames, question, path_tool, args)
            rows.append(row)
            print(f"{row['path']:<10} {question[:48]:<48} Key facts {_fmt(row['facts'])} | first bullet "
                  f"{_fmt(row['first'])} | full {row['full']:.1f} s | {' + '.join(row['models'])}"
                  + (f" (failover: {row['fallback']})" if row["fallback"] else "")
                  + (f" | ERROR {row['error']}" if row["error"] else ""))
    return rows


def _fmt(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    return f"{seconds * 1000:.0f} ms" if seconds < 1 else f"{seconds:.1f} s"


def _avg(rows: list[dict], key: str) -> str:
    values = [r[key] for r in rows if r[key] is not None]
    return f"avg {_fmt(sum(values) / len(values))}, max {_fmt(max(values))}" if values else "-"


def write(rows: list[dict]) -> None:
    lines = [MARKER, "",
             "Measured by `python scripts/measure_fast_path.py`, fresh conversation per question, answer cache not "
             "used, Streamlit rendering time not included. **Card click**: the known tool runs at once (fast path). "
             "**Typed**: the LLM first chooses the tool. Key facts = Answer line, Key facts and Evidence shown; "
             "first bullet = first streamed token of the explanation; full = explanation finished and checked.", "",
             "| question | path | Key facts | first bullet | full answer | model | tokens | checks |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        model = " + ".join(m.split("/")[-1] for m in r["models"]) + (f" (failover: {r['fallback']})" if r["fallback"] else "")
        checks = "error: " + r["error"] if r["error"] else (
            "grounding FAILED" if r["grounding"] else
            f"passed{' after retry' if r['retried'] else ''}, {r['flags']} untraceable numbers")
        lines.append(f"| {r['question']} | {r['path']} | {_fmt(r['facts'])} | {_fmt(r['first'])} | {r['full']:.1f} s "
                     f"| {model} | {r['tokens']:,} | {checks} |")
    lines.append("")
    for path in ("card click", "typed"):
        sub = [r for r in rows if r["path"] == path]
        lines.append(f"- **{path.capitalize()}**: Key facts {_avg(sub, 'facts')}; first bullet {_avg(sub, 'first')}; "
                     f"full answer {_avg(sub, 'full')}.")
    text = RESULTS.read_text(encoding="utf-8") if RESULTS.exists() else "# Agent evaluation results\n"
    text = text.split(MARKER)[0].split("## Fast path (example cards)")[0].rstrip() + "\n\n" + "\n".join(lines) + "\n"
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
