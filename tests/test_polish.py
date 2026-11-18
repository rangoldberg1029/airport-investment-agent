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
    record = execute_tool("rank_airports", {"states": ["MA", "CT"], "weights": {"growth": 0.6}}, frames)
    d = view(record)
    assert list(d)[0] == "headline"
    rows = record.envelope["data"]["ranking"]["ranking"]
    expected = "; ".join(f"{r['rank']}. {r['airport']} {r['airport']} Intl {r['investment_score']}" for r in rows)
    assert d["headline"] == f"Ranking (growth-weighted): {expected}"
    assert [row[2] for row in d["rows"]] == [f"{r['airport']} Intl" for r in rows]      # name column


def test_rank_headline_default_weights_label(frames):
    d = view(execute_tool("rank_airports", {"states": ["MA", "CT"]}, frames))
    assert d["headline"].startswith("Ranking (default weights): 1. ")


@pytest.mark.parametrize("name,args,path", [
    ("get_airport_profile", {"airport": "AAA"}, lambda d: d["name"]),
    ("compare_congestion", {"airports": ["AAA", "BBB"]}, lambda d: d["names"]["AAA"]),
    ("long_haul_share", {"airport": "AAA"}, lambda d: d["name"]),
    ("unmet_demand", {"airport": "AAA"}, lambda d: d["name"]),
])
def test_every_tool_view_has_official_name(frames, name, args, path):
    assert path(view(execute_tool(name, args, frames))) == "AAA Intl"


# ---------------------------------------------------------------- 3. follow-ups

def test_followups_per_tool(frames):
    rank = execute_tool("rank_airports", {"states": ["MA", "CT"]}, frames)
    top = rank.envelope["data"]["ranking"]["ranking"][0]["airport"]
    assert suggestions([rank]) == ["Rank them again with growth weighted higher", f"Show the profile of {top}",
                                   f"What is the unmet demand at {top}?"]
    regrown = execute_tool("rank_airports", {"states": ["MA"], "weights": {"growth": 0.6}}, frames)
    assert suggestions([regrown])[0] == "Rank them again with default weights"

    compare = execute_tool("compare_congestion", {"airports": ["AAA", "BBB"]}, frames)
    rows = sorted(compare.envelope["data"]["congestion"]["airports"], key=lambda r: r["congestion_index"])
    assert suggestions([compare])[0] == f"Why is {rows[0]['airport']} lower?"

    unmet = execute_tool("unmet_demand", {"airport": "AAA"}, frames)
    assert suggestions([unmet])[0] == "What about SFO?"
    assert len(suggestions([unmet])) <= 3


def test_followups_use_last_successful_call_and_skip_errors(frames):
    ok = execute_tool("long_haul_share", {"airport": "AAA"}, frames)
    bad = execute_tool("unmet_demand", {"airport": "Atlantis"}, frames)
    assert suggestions([ok, bad])[0] == "What if long-haul starts at 2,500 miles?"
    assert suggestions([bad]) == [] and suggestions([]) == []


# ---------------------------------------------------------------- 4. readable periods

def test_readable_period():
    assert readable_period("2024-01..2026-08") == "Jan 2024 – Aug 2026"
    assert readable_period("2023..2025") == "2023 – 2025"
    assert readable_period("real time") == "Real time"


def test_model_info_shows_fallback():
    client = type("C", (), {"model": "openai/gpt-oss-120b", "reasoning_effort": "low",
                            "fallback_model": "openai/gpt-oss-20b"})()
    assert setup.model_info(client).endswith("**Fallback:** openai/gpt-oss-20b")


# ---------------------------------------------------------------- 7. daily quota

class Http429(Exception):
    status_code = 429

    def __init__(self, message):
        super().__init__(message)
        self.response = type("R", (), {"headers": {"retry-after": "513"}})()


TPD = Http429("Rate limit reached for model `openai/gpt-oss-120b` on tokens per day (TPD): Limit 200000, Used 199171")
TPM = Http429("Rate limit reached for model `openai/gpt-oss-120b` on tokens per minute (TPM): Limit 8000")


def test_is_daily_limit():
    assert is_daily_limit(TPD) and not is_daily_limit(TPM)


class _Seq:
    def __init__(self, outcomes):
        self.outcomes, self.models = list(outcomes), []
        outer = self

        class _Raw:
            def create(self, **kwargs):
                outer.models.append(kwargs["model"])
                item = outer.outcomes.pop(0)
                if isinstance(item, Exception):
                    raise item
                return item
        self.chat = type("Chat", (), {"completions": type("Comp", (), {"with_raw_response": _Raw()})()})()


def _raw(content="ok"):
    msg = type("M", (), {"content": content, "tool_calls": None})()
    usage = type("U", (), {"total_tokens": 10, "prompt_tokens": 10, "completion_tokens": 0})()
    parsed = type("C", (), {"choices": [type("Ch", (), {"message": msg})()], "usage": usage})()
    return type("R", (), {"headers": {}, "parse": lambda self: parsed})()


def _client():
    from agent.llm import OpenAIChat
    return OpenAIChat(api_key="test", model="openai/gpt-oss-120b", fallback_model="openai/gpt-oss-20b",
                      sleep=lambda s: None)


def test_daily_limit_switches_to_fallback_for_rest_of_session():
    client = _client()
    client.client = _Seq([TPD, _raw("one"), _raw("two")])
    assert client.complete([{"role": "user", "content": "q1"}], tools=[]).model == "openai/gpt-oss-20b"
    assert client.complete([{"role": "user", "content": "q2"}], tools=[]).model == "openai/gpt-oss-20b"
    assert client.client.models == ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "openai/gpt-oss-20b"]
    assert client.exhausted == {"openai/gpt-oss-120b"}


def test_both_models_exhausted_raises_quota_message(frames):
    client = _client()
    client.client = _Seq([TPD, TPD])
    result = Agent(client, frames).ask("Rank New England airports")
    assert result.error == "quota_exhausted" and result.text == QUOTA_MESSAGE
    assert client.exhausted == {"openai/gpt-oss-120b", "openai/gpt-oss-20b"}


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
