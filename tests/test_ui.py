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
    assert "investment score" in text and "Top 3 is" in text
    assert "Why the indices differ (AAA vs BBB" in text and "RAISES operations per runway" in text
    assert "raises AAA by" in text and " is changes " not in text
    assert "23.08% of 156 passenger departures fly 3,000 statute miles or more" in text
    assert "unmet-demand proxy score" in text
    assert "Capacity gap at target load factor 0.85: 12-month window" in text and "peak month 2026-07" in text
    assert "AAA (large hub, MA)" in text


def test_key_facts_are_human_readable(records):
    text = "\n".join(key_facts(list(records.values())))
    for snake in ("lf_avg", "lf_peak", "nas_share", "delay_rate", "growth_yoy", "ops_per_runway", "_pct"):
        assert snake not in text, snake
    assert "Average load factor" in text and "Arrival delay rate" in text


def test_capacity_gap_lines_have_seat_and_departure_numbers(records):
    lines = [f for f in key_facts([records["unmet_demand"]]) if f.startswith("Capacity gap")]
    assert len(lines) == 2
    assert any("seats (+" in line and "departures" in line for line in lines)


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
    text = str(spec)
    assert "_pct" not in text and "lf_avg" not in text and "nas_share" not in text   # readable legends
    assert "'tickCount': 5" in text


CODE_NAME_RE = r"\b[a-z]+_[a-z_]+\b|\b[a-z]+[A-Z][a-zA-Z]+\b"     # snake_case or camelCase identifiers


@pytest.fixture
def live_record(frames):
    from agent.tools import live_airport_status

    def feed():
        return [{"airportId": "AAA", "groundStop": {"impactingCondition": "wind",
                                                    "programExpirationTime": "2026-11-18T20:00:00Z"},
                 "airportConfig": {"arrivalRunwayConfig": "28L (.308)", "arrivalRate": 36}}], "stamp"
    from datetime import date
    today = date(2026, 11, 18)                       # fixed "today": the event time is shown without a date
    return execute_tool("live_airport_status", {"airport": "AAA"}, frames,
                        {"live_airport_status": lambda f, airport: live_airport_status(f, airport, fetch=feed,
                                                                                       today=today)})


def test_evidence_tables_and_charts_have_no_code_names(records, live_record):
    import re
    for record in [*records.values(), live_record]:
        for title, df in evidence.tables(record):
            texts = [title, *map(str, df.columns), *df.astype(str).values.ravel()]
            bad = sorted({m for t in texts for m in re.findall(CODE_NAME_RE, t)})
            assert not bad, (record.name, title, bad)
        ch = evidence.chart(record)
        if ch is not None:
            values = ch.to_dict()
            data = values.get("datasets", {})
            shown = [str(row.get(k)) for rows in data.values() for row in rows for k in ("component", "measure")
                     if row.get(k) is not None]
            assert not [s for s in shown if re.search(CODE_NAME_RE, s)], (record.name, shown)


def test_live_evidence_shows_readable_event_lines(live_record):
    (title, df), = evidence.tables(live_record)
    assert list(df.columns) == ["event"]
    assert df["event"].iloc[0].startswith("Ground stop until 20:00 UTC")
    assert "(.308)" not in " ".join(df["event"])


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
    assert "Manual download" in text and "Public API (Socrata SODA)" in text      # first letter only capitalized
    assert "| ID | Source | Access | Period | Used for |" in text
    assert "Sep 2024 – Aug 2026" in text and "2024-09..2026-08" not in text        # readable periods
    assert "Every number is computed by deterministic Python" in text
    assert "growth-led" in text and "Google Gemini `gemini-3.5-flash-lite`" in text
    assert "Demo using public data. Not investment advice." in text


@pytest.mark.parametrize("secrets,expected_key,voice", [
    ({"GEMINI_API_KEY": "g-secret"}, "g-secret", False),                         # Gemini only: no voice
    ({"GEMINI_API_KEY": "g-secret", "GROQ_API_KEY": "q-secret"}, "g-secret", True),
    ({}, None, False),                                                           # neither: setup card
])
def test_keys_from_streamlit_secrets(monkeypatch, secrets, expected_key, voice):
    """Streamlit Community Cloud: keys come only from st.secrets (no .env, no environment variables)."""
    from ui import setup
    for k in setup.SECRET_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(setup, "load_dotenv", lambda *a, **kw: None)
    monkeypatch.setattr(setup.st, "secrets", secrets)
    assert setup.get_api_key() == expected_key
    assert setup.voice_available() is voice


@pytest.mark.parametrize("keys,expected", [
    ({"GEMINI_API_KEY": "g", "GROQ_API_KEY": "q"}, "Voice input and failover are **on**."),
    ({"GEMINI_API_KEY": "g"}, "Voice input and failover are **off** (no `GROQ_API_KEY`)."),
    ({"GROQ_API_KEY": "q"}, "Running on **Groq only**"),
    ({}, ""),
])
def test_about_panel_describes_the_configured_setup(monkeypatch, keys, expected):
    from ui.render import setup_status
    for k in ("GEMINI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    for k, v in keys.items():
        monkeypatch.setenv(k, v)
    assert setup_status().strip().startswith(expected) if expected else setup_status() == ""
