"""UI polish: ranking headline + official names, follow-up suggestions, readable periods, daily-quota handling."""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from agent.agent import Agent
from agent.llm import QUOTA_MESSAGE, LLMResponse, ToolCall, is_daily_limit
from agent.tools import execute_tool
from ui import setup
from ui.cache import AnswerCache
from ui.followups import suggestions
from ui.render import readable_period

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def view(record) -> dict:
    return json.loads(record.llm_view)["data"]


# ---------------------------------------------------------------- 6. ranking headline + names

def test_rank_headline_in_exact_tool_order_with_official_names(frames):
    record = execute_tool("rank_airports", {"states": ["MA", "CT"], "preset": "growth-led"}, frames)
    d = view(record)
    assert list(d)[0] == "headline"
    rows = record.envelope["data"]["ranking"]["ranking"]
    expected = "; ".join(f"{r['rank']}. {r['airport']} {r['airport']} Intl {r['investment_score']}" for r in rows)
    assert d["headline"] == ("Ranking with preset growth-led (demand pressure 25%, growth 45%, congestion 15%, "
                             f"scale 15%): {expected}")
    assert d["preset"] == "growth-led"
    assert [row[2] for row in d["rows"]] == [f"{r['airport']} Intl" for r in rows]      # name column


def test_rank_headline_default_and_custom_weights_labels(frames):
    d = view(execute_tool("rank_airports", {"states": ["MA", "CT"]}, frames))
    assert d["headline"].startswith("Ranking with preset balanced (default) (demand pressure 35%, growth 25%, "
                                    "congestion 25%, scale 15%): 1. ")
    custom = view(execute_tool("rank_airports", {"states": ["MA", "CT"], "weights": {"growth": 0.6}}, frames))
    assert custom["headline"].startswith("Ranking with custom weights (") and custom["preset"] == "custom"


@pytest.mark.parametrize("name,args,path", [
    ("get_airport_profile", {"airport": "AAA"}, lambda d: d["name"]),
    ("compare_congestion", {"airports": ["AAA", "BBB"]}, lambda d: d["names"]["AAA"]),
    ("long_haul_share", {"airport": "AAA"}, lambda d: d["name"]),
    ("unmet_demand", {"airport": "AAA"}, lambda d: d["name"]),
])
def test_every_tool_view_has_official_name(frames, name, args, path):
    assert path(view(execute_tool(name, args, frames))) == "AAA Intl"


# ---------------------------------------------------------------- 3. follow-ups

def questions(records) -> list[str]:
    return [s.question for s in suggestions(records)]


def test_followups_per_tool(frames):
    rank = execute_tool("rank_airports", {"states": ["MA", "CT"]}, frames)
    top = rank.envelope["data"]["ranking"]["ranking"][0]["airport"]
    assert questions([rank]) == ["Rank them again with growth weighted higher", f"Show the profile of {top}",
                                 f"What is the unmet demand at {top}?"]
    regrown = execute_tool("rank_airports", {"states": ["MA"], "preset": "growth-led"}, frames)
    assert questions([regrown])[0] == "Rank them again with default weights"

    compare = execute_tool("compare_congestion", {"airports": ["AAA", "BBB"]}, frames)
    rows = sorted(compare.envelope["data"]["congestion"]["airports"], key=lambda r: r["congestion_index"])
    assert questions([compare])[0] == f"Why is {rows[0]['airport']} lower?"

    unmet = execute_tool("unmet_demand", {"airport": "AAA"}, frames)
    assert questions([unmet])[0] == "What about SFO?"
    assert len(suggestions([unmet])) <= 3


def test_followups_carry_the_tool_call_that_answers_them(frames):
    rank = execute_tool("rank_airports", {"region": "new england", "top_n": 5}, frames)
    again, profile, unmet = suggestions([rank])
    assert (again.tool, again.args) == ("rank_airports", {"region": "new england", "preset": "growth-led"})
    top = rank.envelope["data"]["ranking"]["ranking"][0]["airport"]
    assert (profile.tool, profile.args) == ("get_airport_profile", {"airport": top})
    assert (unmet.tool, unmet.args) == ("unmet_demand", {"airport": top})
    # re-ranking with these arguments uses the growth-led preset weights
    regrown = execute_tool(again.tool, again.args, frames)
    assert regrown.envelope["data"]["ranking"]["weights"]["growth"] == 0.45
    assert regrown.envelope["data"]["ranking"]["preset"] == "growth-led"
    assert suggestions([regrown])[0].args == {"region": "new england"}          # back to default weights

    compare = execute_tool("compare_congestion", {"airports": ["AAA", "BBB"]}, frames)
    why = suggestions([compare])[0]
    assert (why.tool, why.args) == ("compare_congestion", {"airports": ["AAA", "BBB"]})
    long_haul = suggestions([execute_tool("long_haul_share", {"airport": "AAA"}, frames)])[0]
    assert (long_haul.tool, long_haul.args) == ("long_haul_share", {"airport": "AAA", "threshold_mi": 2500})


def test_followups_use_last_successful_call_and_skip_errors(frames):
    ok = execute_tool("long_haul_share", {"airport": "AAA"}, frames)
    bad = execute_tool("unmet_demand", {"airport": "Atlantis"}, frames)
    assert questions([ok, bad])[0] == "What if long-haul starts at 2,500 miles?"
    assert suggestions([bad]) == [] and suggestions([]) == []


# ---------------------------------------------------------------- 4. readable periods

def test_readable_period():
    assert readable_period("2024-01..2026-08") == "Jan 2024 – Aug 2026"
    assert readable_period("2023..2025") == "2023 – 2025"
    assert readable_period("real time") == "Real time"


def test_model_info_shows_failover_per_mode():
    from tests.test_llm import _Seq, client_with, gemini, groq
    both = setup.model_info(client_with(gemini(_Seq([])), groq(_Seq([]))))
    assert both.startswith("gemini-3.5-flash-lite · generativelanguage.googleapis.com · reasoning: minimal")
    assert both.endswith("**Failover:** openai/gpt-oss-120b · api.groq.com")
    assert setup.model_info(client_with(gemini(_Seq([])))).endswith("**Failover:** off (set GROQ_API_KEY)")
    groq_only = setup.model_info(client_with(groq(_Seq([]))))
    assert groq_only.startswith("openai/gpt-oss-120b · api.groq.com · reasoning: low")
    assert "Groq only" in groq_only


# ---------------------------------------------------------------- 7. daily quota

class Http429(Exception):
    status_code = 429

    def __init__(self, message):
        super().__init__(message)
        self.response = type("R", (), {"headers": {"retry-after": "513"}})()


TPD = Http429("Rate limit reached for model `openai/gpt-oss-120b` on tokens per day (TPD): Limit 200000, Used 199171")
TPM = Http429("Rate limit reached for model `openai/gpt-oss-120b` on tokens per minute (TPM): Limit 8000")
GEMINI_RPD = Http429("Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, "
                     "limit: 1000 ... quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")
GEMINI_RPM = Http429("Quota exceeded ... quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier")


def test_is_daily_limit():
    assert is_daily_limit(TPD) and not is_daily_limit(TPM)
    assert is_daily_limit(GEMINI_RPD) and not is_daily_limit(GEMINI_RPM)


def _clients(gemini_outcomes, groq_outcomes):
    from tests.test_llm import _Seq, client_with, gemini, groq
    g, q = _Seq(gemini_outcomes), _Seq(groq_outcomes)
    return client_with(gemini(g), groq(q)), g, q


def test_daily_limit_switches_to_fallback_for_rest_of_session():
    from tests.test_llm import _raw
    client, g, q = _clients([GEMINI_RPD], [_raw("one"), _raw("two")])
    assert client.complete([{"role": "user", "content": "q1"}], tools=[]).model == "openai/gpt-oss-120b"
    assert client.complete([{"role": "user", "content": "q2"}], tools=[]).model == "openai/gpt-oss-120b"
    assert g.models == ["gemini-3.5-flash-lite"] and q.models == ["openai/gpt-oss-120b"] * 2
    assert client.exhausted == {"gemini-3.5-flash-lite"}


def test_both_models_exhausted_raises_quota_message(frames):
    client, _, _ = _clients([GEMINI_RPD], [TPD])
    result = Agent(client, frames).ask("Rank New England airports")
    assert result.error == "quota_exhausted" and result.text == QUOTA_MESSAGE
    assert client.exhausted == {"gemini-3.5-flash-lite", "openai/gpt-oss-120b"}


# ---------------------------------------------------------------- 1, 3, 7 in the app

class AppLLM:
    """rank_airports for MA/CT, then a fixed answer. Or: always quota-exhausted."""

    model = "fake-model"
    reasoning_effort = None

    def __init__(self, exhausted: bool = False):
        self.exhausted = exhausted
        self.calls = 0

    def set_sleep(self, sleep):
        pass

    def complete(self, messages, tools, tool_choice="auto"):
        from agent.llm import QuotaExhausted
        if self.exhausted:
            raise QuotaExhausted(QUOTA_MESSAGE)
        self.calls += 1
        if messages[-1]["role"] == "user":
            return LLMResponse("", [ToolCall(f"c{self.calls}", "rank_airports", '{"states": ["MA", "CT"]}')], 10)
        return LLMResponse("Ranked.", [], 10)


@pytest.fixture
def app(monkeypatch, frames):
    def configure(client):
        monkeypatch.setattr(setup, "get_api_key", lambda: "test-key")
        monkeypatch.setattr(setup, "get_frames", lambda: frames)
        monkeypatch.setattr(setup, "data_info", lambda: {"sources": [], "built": "2026-11-18"})
        monkeypatch.setattr(setup, "make_client", lambda: client)
        cache = AnswerCache()
        monkeypatch.setattr(setup, "get_answer_cache", lambda: cache)
        at = AppTest.from_file(APP, default_timeout=60)
        at.run()
        return at
    return configure


def test_empty_state_cards_then_disappear(app):
    at = app(AppLLM())
    assert any("Welcome" in m.value for m in at.markdown)
    cards = sorted(b.key for b in at.button if b.key and b.key.startswith("card_"))   # 2 columns: column order
    assert cards == [f"card_{i}" for i in range(4)]
    at.button(key="card_0").click().run()
    assert not at.exception
    assert not any("Welcome" in m.value for m in at.markdown)               # gone once a conversation exists
    assert at.session_state.messages[0]["text"].startswith("Which airports in New England")


def test_followup_button_sends_question(app):
    llm = AppLLM()
    at = app(llm)
    at.button(key="card_0").click().run()
    followups = [b for b in at.button if b.key and "_followup_" in b.key]
    assert followups and followups[0].label == "Rank them again with growth weighted higher"
    followups[0].click().run()
    assert not at.exception
    assert at.session_state.messages[-2]["text"] == "Rank them again with growth weighted higher"


def test_quota_exhausted_message_in_app(app):
    at = app(AppLLM(exhausted=True))
    at.button(key="card_0").click().run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "daily capacity has been reached" in text
    assert not [b for b in at.button if b.key and "_followup_" in b.key]     # no follow-ups on an error
