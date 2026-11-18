"""Pure UI helpers: key facts, evidence tables/charts, progress labels and countdown (no Streamlit server)."""

import pandas as pd
import pytest

from agent.tools import execute_tool
from ui import evidence
from ui.key_facts import key_facts
from ui.progress import ANALYZING, COMPOSING, StatusReporter, tool_label, wait_message
from ui.render import about_markdown

CALLS = [
    ("rank_airports", {"states": ["MA", "CT"]}),
    ("compare_congestion", {"airports": ["AAA", "BBB"]}),
    ("long_haul_share", {"airport": "AAA"}),
    ("unmet_demand", {"airport": "AAA"}),
    ("get_airport_profile", {"airport": "AAA"}),
]


@pytest.fixture
def records(frames):
    return {name: execute_tool(name, args, frames) for name, args in CALLS}


def test_key_facts_come_from_tool_outputs(records):
    facts = key_facts(list(records.values()))
    text = "\n".join(facts)
    assert "investment score" in text and "Top 3 across weight presets" in text
    assert "has the higher congestion index" in text and "RAISES operations per runway" in text
    assert "23.08% of 156 passenger departures fly >= 3,000 statute miles" in text
    assert "unmet-demand proxy score" in text and "Target LF 0.85" in text
    assert "AAA (large hub, MA)" in text


def test_key_facts_report_errors_without_raising(frames):
    bad = execute_tool("unmet_demand", {"airport": "Atlantis"}, frames)
    assert key_facts([bad]) == [f"Tool unmet_demand returned an error: {bad.error}"]


@pytest.mark.parametrize("name", [n for n, _ in CALLS])
def test_evidence_tables_are_dataframes(records, name):
    tables = evidence.tables(records[name])
    assert tables and all(isinstance(df, pd.DataFrame) and not df.empty for _, df in tables)


def test_evidence_numeric_columns(records):
    rank = dict(evidence.tables(records["rank_airports"]))["Investment score ranking"]
    assert pd.api.types.is_numeric_dtype(rank["investment score"])
    lh = dict(evidence.tables(records["long_haul_share"]))["Long-haul share by distance threshold"]
    assert list(lh["threshold (mi)"]) == [2500, 3000, 4000]


@pytest.mark.parametrize("name", ["rank_airports", "compare_congestion", "unmet_demand", "long_haul_share"])
def test_charts_build(records, name):
    spec = evidence.chart(records[name]).to_dict()
    assert spec["mark"]["type"] == "bar"


def test_no_chart_for_profile_or_errors(records, frames):
    assert evidence.chart(records["get_airport_profile"]) is None
    assert evidence.chart(execute_tool("unmet_demand", {"airport": "Atlantis"}, frames)) is None


def test_tool_label():
    assert tool_label("compare_congestion", '{"airports": ["LAX", "SNA"]}') == \
        "Running tool: compare_congestion (LAX, SNA)"
    assert tool_label("rank_airports", {"region": "new england", "weights": {"growth": 0.5}}) == \
        "Running tool: rank_airports (new england; growth=0.5)"
    assert tool_label("long_haul_share", "{bad json") == "Running tool: long_haul_share"


class FakeStatus:
    def __init__(self):
        self.labels, self.lines, self.notices = [], [], []

    def update(self, label=None, **kw):
        self.labels.append(label)

    def write(self, text):
        self.lines.append(text)

    def empty(self):
        status = self

        class Slot:
            def info(self, text):
                status.notices.append(text)

            def empty(self):
                status.notices.append("<cleared>")
        return Slot()


def test_status_reporter_steps_and_countdown():
    status, now = FakeStatus(), [0.0]

    def sleep(s):
        now[0] += s

    rep = StatusReporter(status, clock=lambda: now[0], sleep=sleep)
    rep.on_event("llm_call", round=0, after_tools=False)
    rep.on_event("tool_start", name="unmet_demand", args='{"airport": "SFO"}')
    rep.on_event("llm_call", round=1, after_tools=True)
    assert status.labels == [ANALYZING, "Running tool: unmet_demand (SFO)", COMPOSING]
    rep.sleep(2.5)
    assert status.notices == [wait_message(3), wait_message(2), wait_message(1), "<cleared>"]
    assert "free" not in wait_message(5).lower()


def test_about_text_is_generated_from_config(frames):
    text = about_markdown(frames)
    assert "Sep 2025 - Aug 2026" in text and "3,000" in text and "5,000 ft" in text
    assert "manual download" in text and "public API" in text
    assert "Every number is computed by deterministic Python" in text
