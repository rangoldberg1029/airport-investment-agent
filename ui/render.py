"""Rendering: the 'About this agent' text (pure), one assistant answer in a fixed frame (stored or live)."""

import json
import os
import re
from dataclasses import dataclass
from datetime import date

import streamlit as st

from agent.llm import DEFAULT_GEMINI_MODEL, DEFAULT_GROQ_MODEL
from agent.tools import ToolCallRecord, _source_inventory
from data.repository import Frames
from scoring.common import window_label
from scoring.config import (
    DEFAULT_LONG_HAUL_MI, DEFAULT_PRESET, LOAD_FACTOR_SENSITIVITY, LONG_HAUL_THRESHOLDS_MI,
    MIN_COMMERCIAL_RUNWAY_FT, PRIOR_WINDOW, RANKING_UNIVERSE_HUBS, TARGET_LOAD_FACTOR, WEIGHT_PRESETS, WINDOW,
)
from ui import evidence
from ui.followups import suggestions
from ui.key_facts import key_facts

EXAMPLES = [
    "Which airports in New England are strong candidates for terminal expansion?",
    "Compare LA (LAX) and Santa Ana (SNA) airport congestion levels.",
    "What is the percentage of long-haul flights out of Anchorage (ANC)?",
    "What is the unmet flight demand at SFO and why?",
]
# The tool call that answers each example (fast path: run it directly, then one LLM call explains it).
EXAMPLE_CALLS = [
    ("rank_airports", {"region": "new england"}),
    ("compare_congestion", {"airports": ["LAX", "SNA"]}),
    ("long_haul_share", {"airport": "ANC"}),
    ("unmet_demand", {"airport": "SFO"}),
]


def cap(text: str) -> str:
    """Upper-case the first letter only ('public API' -> 'Public API', unlike str.capitalize)."""
    return text[:1].upper() + text[1:]


def readable_period(span: str) -> str:
    """'2024-01..2026-08' -> 'Jan 2024 – Aug 2026'; '2023..2025' -> '2023 – 2025'; anything else unchanged."""
    months = re.fullmatch(r"(\d{4})-(\d{2})\.\.(\d{4})-(\d{2})", span or "")
    if months:
        y0, m0, y1, m1 = map(int, months.groups())
        return f"{date(y0, m0, 1):%b %Y} – {date(y1, m1, 1):%b %Y}"
    years = re.fullmatch(r"(\d{4})\.\.(\d{4})", span or "")
    if years:
        return f"{years.group(1)} – {years.group(2)}"
    return cap(span or "")


def setup_status() -> str:
    """Which of the two keys are configured, in one sentence (empty when neither is: the setup card shows)."""
    gemini_key, groq_key = bool(os.getenv("GEMINI_API_KEY")), bool(os.getenv("GROQ_API_KEY"))
    if gemini_key and groq_key:
        return " Voice input and failover are **on**."
    if gemini_key:
        return " Voice input and failover are **off** (no `GROQ_API_KEY`)."
    if groq_key:
        return " Running on **Groq only** (no `GEMINI_API_KEY`): voice input on, no failover."
    return ""


def models_markdown() -> str:
    """The language-model setup, from agent/llm.py constants and the configured environment."""
    gemini = os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
    groq = os.getenv("GROQ_MODEL") or DEFAULT_GROQ_MODEL
    status = setup_status()
    return (f"**Language models.** Google Gemini `{gemini}` (free tier, `GEMINI_API_KEY`). Optional Groq "
            f"(free tier, `GROQ_API_KEY`) adds voice input (Whisper) and automatic failover to `{groq}`: "
            f"immediately when Gemini is rate-limited, and after one retry when Gemini returns a server error or "
            f"times out. The status line under each answer names the model that answered.{status}")


def about_markdown(frames: Frames) -> str:
    """'About this agent' content, generated from config and the loaded data so it never goes stale."""
    sources = "\n".join(f"| {s['id']} | {s['name']} | {cap(s['access'])} | {readable_period(s['period'])} | "
                        f"{cap(', '.join(s['feeds']))} |" for s in _source_inventory(frames))
    presets = ", ".join(f"{n} (default)" if n == DEFAULT_PRESET else n for n in WEIGHT_PRESETS)
    return f"""
**What it does.** Ranks and explains US airports where terminal or capacity investment is most likely to pay off,
based on passenger demand versus capacity: load factors, growth, delays, airspace (NAS) delays and runway use.

**Scope.** US commercial airports, passenger service only (all-cargo flights excluded). Standard window
**{window_label(WINDOW)}** (trailing 12 months); growth is measured against **{window_label(PRIOR_WINDOW)}**.
Out of scope: non-US airports, construction costs, financing and returns, fares, forecasts.

**Key assumptions** (all configurable in `scoring/config.py`)
- Rankings compare FAA {'/'.join(RANKING_UNIVERSE_HUBS)} hubs nationally (percentiles, 0-100).
- Investment-score weights come from named presets: {presets}. A request such as "weight growth higher" uses
  a preset, so the same question always gives the same ranking; custom weights only when you give numbers.
- Long-haul = segment distance >= {DEFAULT_LONG_HAUL_MI:,} statute miles (sensitivity at
  {', '.join(f'{t:,}' for t in LONG_HAUL_THRESHOLDS_MI)} mi).
- Capacity gap target load factor {TARGET_LOAD_FACTOR:.2f} (sensitivity {', '.join(f'{x:.2f}' for x in LOAD_FACTOR_SENSITIVITY)}).
- Only runways >= {MIN_COMMERCIAL_RUNWAY_FT:,} ft count as commercial runways.
- Unmet demand is a proxy: true spilled demand is not observable in public data.

**How AI is used.** The language model only (1) understands the question, (2) chooses a tool and its arguments and
(3) explains the tool output. **Every number is computed by deterministic Python.** The *Answer* line of each reply
is built by code from the tool output; the model writes only the *Why / drivers* bullets, which are checked:
numbers that cannot be traced to a tool output are flagged. *Key facts* and *Evidence* under each answer are
computed, not AI-generated.

{models_markdown()}

**Data sources**

| ID | Source | Access | Period | Used for |
|---|---|---|---|---|
{sources}
"""


def _fmt_args(args: dict) -> str:
    return json.dumps(args, ensure_ascii=False)


def render_evidence(records: list[ToolCallRecord]) -> None:
    for i, r in enumerate(records):
        if i:
            st.divider()
        st.markdown(f"**Tool:** `{r.name}`  \n**Arguments:** `{_fmt_args(r.args)}`")
        if r.error:
            st.error(r.error)
            continue
        ch = evidence.chart(r)
        if ch is not None:
            st.altair_chart(ch, width="stretch")
        tables = evidence.tables(r)
        for title, df in tables:
            st.caption(title)
            st.dataframe(df, hide_index=True, width="stretch")
        if not tables and r.name == "explain_methodology":
            st.json(r.envelope["data"], expanded=False)
        env = r.envelope
        st.caption(f"**Data period:** {env['data_period']}")
        if env["sources"]:
            st.caption("**Sources:** " + "; ".join(env["sources"]))
        if env["assumptions"]:
            st.caption("**Assumptions:**\n" + "\n".join(f"- {a}" for a in env["assumptions"]))
        if env["caveats"]:
            st.caption("**Caveats:**\n" + "\n".join(f"- {c}" for c in env["caveats"]))


def render_followups(records: list[ToolCallRecord], key: str, on_followup, asked=()) -> None:
    """Up to 3 suggested follow-up questions as small buttons, none already asked in this conversation (`asked`);
    a click calls on_followup(question, tool, args)."""
    items = suggestions(records, asked)
    if not items:
        return
    cols = st.columns(len(items))
    for i, (col, s) in enumerate(zip(cols, items)):
        col.button(s.question, key=f"{key}_followup_{i}", type="tertiary", icon=":material/subdirectory_arrow_right:",
                   on_click=on_followup, args=(s.question, s.tool, s.args))


def render_details(records: list[ToolCallRecord]) -> None:
    """'Key facts' (computed) and the Evidence expander for one answer."""
    facts = key_facts(records)
    if facts:
        with st.container(border=True):
            st.markdown("**Key facts** · *computed, not AI-generated*")
            st.markdown("\n".join(f"- {f}" for f in facts))
    if records:
        with st.expander(f"Evidence ({len(records)} tool call{'s' if len(records) != 1 else ''})"):
            render_evidence(records)


@dataclass
class AnswerFrame:
    """The fixed element layout of one assistant answer: status, Answer line, Why / drivers, Key facts +
    Evidence, follow-ups. History and live answers use the same slots in the same order, so when the page reruns
    each slot is replaced in place (no greyed duplicates of cards or follow-up buttons)."""
    status: object
    headline: object
    why: object
    details: object
    followups: object

    @classmethod
    def create(cls) -> "AnswerFrame":
        return cls(st.empty(), st.empty(), st.empty(), st.empty(), st.empty())


WRITING = "Writing explanation…"
SPINNER_HTML = (
    '<div style="display:flex;align-items:center;gap:.5rem;opacity:.75;font-size:.9rem">'
    '<span class="aia-spin"></span>' + WRITING + '</div>'
    "<style>.aia-spin{width:14px;height:14px;border:2px solid rgba(31,58,95,.25);border-top-color:#1f3a5f;"
    "border-radius:50%;display:inline-block;animation:aia-spin .8s linear infinite}"
    "@keyframes aia-spin{to{transform:rotate(360deg)}}</style>")


def why_markdown(message: dict) -> str:
    """The Why / drivers part of a stored answer (or its plain reply / error note)."""
    if message.get("headline") and not message.get("error"):
        return "**Why / drivers**\n" + (message.get("explanation") or "")
    return message.get("explanation") or (message.get("text") if not message.get("headline") else "")


def render_assistant(message: dict, frame: AnswerFrame | None = None, key: str | None = None,
                     on_followup=None, asked=()) -> None:
    """A stored answer in its frame: status line, computed Answer line, Why / drivers + check notices, Key facts
    and Evidence, and follow-up buttons (only when `key` and `on_followup` are given: the latest answer; never a
    question in `asked`)."""
    frame = frame or AnswerFrame.create()
    with frame.status.container():
        if message.get("cached"):
            st.badge("Cached result", icon=":material/bolt:", color="gray")
        if message.get("meta"):
            st.caption(message["meta"])
    if message.get("headline"):
        frame.headline.markdown(f"**Answer** - {message['headline']}")
    else:
        frame.headline.empty()
    with frame.why.container():
        st.markdown(why_markdown(message))
        render_checks(message)
    records = message.get("records") or []
    with frame.details.container():
        render_details(records)
    if key is not None and on_followup is not None and not message.get("error"):
        with frame.followups.container():
            render_followups(records, key, on_followup, asked)
    else:
        frame.followups.empty()


class LiveAnswer:
    """Fills an AnswerFrame while a turn runs: Answer line + Key facts + Evidence as soon as the tools ran, a
    spinner where the bullets will appear, then the bullets token by token. On a grounding retry the bullets
    are cleared and streamed again."""

    def __init__(self, frame: AnswerFrame):
        self.frame = frame
        self.headline = ""
        self.text = ""

    def show_facts(self, records: list[ToolCallRecord], headline: str) -> None:
        self.headline = headline
        if headline:
            self.frame.headline.markdown(f"**Answer** - {headline}")
        with self.frame.details.container():
            render_details(records)
        self.waiting()

    def waiting(self) -> None:
        self.text = ""
        self.frame.why.markdown(SPINNER_HTML, unsafe_allow_html=True)

    def token(self, piece: str) -> None:
        self.text += piece
        self.frame.why.markdown(("**Why / drivers**\n" if self.headline else "") + self.text + " ▌")

    def on_event(self, event: str, **kw) -> None:
        if event == "facts_ready":
            self.show_facts(kw["records"], kw.get("headline", ""))
        elif event == "token":
            self.token(kw["text"])
        elif event == "retry":
            self.waiting()

    def finish(self, message: dict) -> None:
        """Replace the streamed text with the final answer (checked bullets + notices)."""
        render_assistant_parts(self.frame, message)


def render_assistant_parts(frame: AnswerFrame, message: dict) -> None:
    """Final Answer line, Why / drivers and notices, Key facts and Evidence (not status or follow-ups)."""
    if message.get("headline"):
        frame.headline.markdown(f"**Answer** - {message['headline']}")
    with frame.why.container():
        st.markdown(why_markdown(message))
        render_checks(message)
    with frame.details.container():
        render_details(message.get("records") or [])


def render_checks(message: dict) -> None:
    """Grounding-check and number-check notices for the LLM-written part of an answer."""
    if message.get("grounding_issues"):
        items = "\n".join(f"- {i}" for i in message["grounding_issues"])
        st.warning("**Grounding check failed:** this answer could not be verified against the tool calls of this "
                   "turn, even after an automatic retry. Treat it with caution and check the Key facts and "
                   "Evidence.\n\n" + items)
    elif message.get("retried"):
        st.caption("A first draft failed the grounding check; this answer comes from an automatic retry "
                   "with a tool call.")
    if message.get("untraceable"):
        items = "\n".join(f"- `{n['value']}` in: ...{n['context']}..." for n in message["untraceable"])
        st.warning("**Number check:** these numbers in the answer could not be traced to a tool output. "
                   "Verify them against the Evidence below.\n\n" + items)
