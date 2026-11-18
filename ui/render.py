"""Rendering: the 'About this agent' text (pure) and one assistant message (answer, key facts, evidence)."""

import json

import streamlit as st

from agent.tools import ToolCallRecord, _source_inventory
from data.repository import Frames
from scoring.common import window_label
from scoring.config import (
    DEFAULT_LONG_HAUL_MI, LOAD_FACTOR_SENSITIVITY, LONG_HAUL_THRESHOLDS_MI, MIN_COMMERCIAL_RUNWAY_FT,
    PRIOR_WINDOW, RANKING_UNIVERSE_HUBS, TARGET_LOAD_FACTOR, WINDOW,
)
from ui import evidence
from ui.key_facts import key_facts

EXAMPLES = [
    "Which airports in New England are strong candidates for terminal expansion?",
    "Compare LA (LAX) and Santa Ana (SNA) airport congestion levels.",
    "What is the percentage of long-haul flights out of Anchorage (ANC)?",
    "What is the unmet flight demand at SFO and why?",
]


def about_markdown(frames: Frames) -> str:
    """'About this agent' content, generated from config and the loaded data so it never goes stale."""
    sources = "\n".join(f"| {s['id']} | {s['name']} | {s['access']} | {s['period']} | {', '.join(s['feeds'])} |"
                        for s in _source_inventory(frames))
    return f"""
**What it does.** Ranks and explains US airports where terminal or capacity investment is most likely to pay off,
based on passenger demand versus capacity: load factors, growth, delays, airspace (NAS) delays and runway use.

**Scope.** US commercial airports, passenger service only (all-cargo flights excluded). Standard window
**{window_label(WINDOW)}** (trailing 12 months); growth is measured against **{window_label(PRIOR_WINDOW)}**.
Out of scope: non-US airports, construction costs, financing and returns, fares, forecasts.

**Key assumptions** (all configurable in `scoring/config.py`)
- Rankings compare FAA {'/'.join(RANKING_UNIVERSE_HUBS)} hubs nationally (percentiles, 0-100).
- Long-haul = segment distance >= {DEFAULT_LONG_HAUL_MI:,} statute miles (sensitivity at
  {', '.join(f'{t:,}' for t in LONG_HAUL_THRESHOLDS_MI)} mi).
- Capacity gap target load factor {TARGET_LOAD_FACTOR:.2f} (sensitivity {', '.join(f'{x:.2f}' for x in LOAD_FACTOR_SENSITIVITY)}).
- Only runways >= {MIN_COMMERCIAL_RUNWAY_FT:,} ft count as commercial runways.
- Unmet demand is a proxy: true spilled demand is not observable in public data.

**How AI is used.** The language model only (1) understands the question, (2) chooses a tool and its arguments and
(3) explains the tool output. **Every number is computed by deterministic Python**, shown in *Key facts* and
*Evidence* under each answer, and checked: numbers in the answer that cannot be traced to a tool output are flagged.

**Data sources**

| id | source | access | period | used for |
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


def render_assistant(message: dict) -> None:
    """Answer text, then 'Key facts' (computed), number-check warning, and the Evidence expander."""
    st.markdown(message["text"])
    records = message.get("records") or []
    facts = key_facts(records)
    if facts:
        with st.container(border=True):
            st.markdown("**Key facts** · *computed, not AI-generated*")
            st.markdown("\n".join(f"- {f}" for f in facts))
    if message.get("untraceable"):
        items = "\n".join(f"- `{n['value']}` in: ...{n['context']}..." for n in message["untraceable"])
        st.warning("**Number check:** these numbers in the answer could not be traced to a tool output. "
                   "Verify them against the Evidence below.\n\n" + items)
    if records:
        with st.expander(f"Evidence ({len(records)} tool call{'s' if len(records) != 1 else ''})"):
            render_evidence(records)
    meta = message.get("meta")
    if meta:
        st.caption(meta)
