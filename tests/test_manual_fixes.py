"""Regression tests for the fixes from manual testing (fast path, guardrails, rendering, answer quality,
live status, streaming, voice). No network."""

from datetime import date
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from agent.agent import Agent
from agent.guardrails import grounding_issues
from agent.llm import OpenAIChat, make_provider
from agent.tools import TOOLS, execute_tool
from data.sources.nas_status import describe_event, describe_events, readable_reason, readable_time
from tests.test_agent import FakeLLM, call, text
from ui import setup
from ui.cache import AnswerCache
from ui.key_facts import key_facts

APP = str(Path(__file__).resolve().parents[1] / "app.py")


# ---------------------------------------------------------------- A1 fast path counts its tool call

def test_fast_path_answer_mentioning_other_topics_passes_grounding(frames):
    """Bullets of a ranking answer may say 'unmet demand' or 'compare congestion' without claiming those tools."""
    fake = FakeLLM([text("- Demand pressure raises CCC; unmet demand is not part of this score. "
                         "To compare congestion, ask for a comparison.")])
    result = Agent(fake, frames).ask_direct("Rank MA and CT", "rank_airports", {"states": ["MA", "CT"]})
    assert result.grounding_issues == [] and not result.retried and fake.tool_choices == ["none"]


def test_explicit_tool_claims_are_still_caught():
    issues = grounding_issues("I used the unmet demand tool.", "q", ["rank_airports"], list(TOOLS), "", set())
    assert issues and "unmet_demand" in issues[0]
    assert grounding_issues("Used rank_airports.", "q", [], list(TOOLS), "", set())


# ---------------------------------------------------------------- A2 out-of-scope refusals

def test_scope_refusal_passes_without_a_tool_call(frames):
    fake = FakeLLM([text("Construction costs are out of scope: this agent covers US airport demand only.")])
    result = Agent(fake, frames).ask("Compare construction costs at LAX and SNA")
    assert result.grounding_issues == [] and not result.retried and len(fake.tool_choices) == 1


def test_non_us_airport_gets_a_scope_answer(frames):
    fake = FakeLLM([call("unmet_demand", {"airport": "LHR"}), text("Heathrow is outside the US scope.")])
    result = Agent(fake, frames).ask("What is the unmet demand at LHR?")
    assert result.headline.startswith("LHR London Heathrow Airport is outside the US scope")
    assert "unknown airport" not in result.text


# ---------------------------------------------------------------- A3 answer cache

class _Client:
    """Fast-path client whose explanation is decided per test."""
    model, reasoning_effort = "fake-model", None

    def __init__(self, answers):
        self.answers = list(answers)

    def set_sleep(self, sleep):
        pass

    def complete(self, messages, tools, tool_choice="auto", on_token=None):
        from agent.llm import LLMResponse
        content = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if on_token:
            on_token(content)
        return LLMResponse(content, [], 10)


@pytest.fixture
def app(monkeypatch, frames):
    def configure(client, voice=True):
        monkeypatch.setattr(setup, "get_api_key", lambda: "test-key")
        monkeypatch.setattr(setup, "voice_available", lambda: voice)
        monkeypatch.setattr(setup, "get_frames", lambda: frames)
        monkeypatch.setattr(setup, "data_info", lambda: {"sources": [], "built": "2026-11-18"})
        monkeypatch.setattr(setup, "make_client", lambda: client)
        cache = AnswerCache()
        monkeypatch.setattr(setup, "get_answer_cache", lambda: cache)
        at = AppTest.from_file(APP, default_timeout=60)
        at.run()
        return at, cache
    return configure


def test_retried_answers_are_never_cached(app):
    # first draft names an uncalled tool -> retry -> clean answer; still not cached because it needed a retry
    at, cache = app(_Client(["- Used the unmet_demand tool.", "- CCC leads."]))
    at.button(key="card_0").click().run()
    assert not at.exception and at.session_state.messages[-1]["retried"]
    assert len(cache) == 0


def test_clean_first_answer_is_cached_and_key_includes_prompt_version(app):
    from agent.prompts import PROMPT_VERSION
    at, cache = app(_Client(["- CCC leads."]))
    at.button(key="card_0").click().run()
    assert len(cache) == 1 and PROMPT_VERSION in next(iter(cache._items))


# ---------------------------------------------------------------- B4 / B5 rendering

def test_cards_hidden_once_a_question_is_submitted_from_any_path(app):
    at, _ = app(_Client(["- CCC leads."]))
    at.chat_input[0].set_value("Rank MA and CT airports").run()                    # typed
    assert not [b for b in at.button if b.key and b.key.startswith("card_")]
    at, _ = app(_Client(["- CCC leads."]))
    at.session_state["voice_text"] = "Rank MA and CT airports"                      # voice transcript
    at.run()
    at.sidebar.button(key="voice_send").click().run()
    assert not at.exception
    assert at.session_state.messages[0]["text"] == "Rank MA and CT airports"
    assert not [b for b in at.button if b.key and b.key.startswith("card_")]
    assert at.session_state["voice_text"] == ""                                     # F20: transcript cleared


def test_follow_up_buttons_only_under_the_latest_answer(app):
    at, _ = app(_Client(["- CCC leads."]))
    at.button(key="card_0").click().run()
    first = [b.key for b in at.button if b.key and "_followup_" in b.key]
    next(b for b in at.button if b.key == first[0]).click().run()                  # ask a follow-up
    keys = [b.key for b in at.button if b.key and "_followup_" in b.key]
    assert keys and all(k.startswith("msg3_") for k in keys)                        # only the 2nd answer (index 3)
    assert len(keys) == len(set(keys))


# ---------------------------------------------------------------- C11 readable labels in the UI

def test_key_facts_have_no_snake_case(frames):
    records = [execute_tool("rank_airports", {"states": ["MA", "CT"], "preset": "growth-led"}, frames),
               execute_tool("compare_congestion", {"airports": ["AAA", "BBB"]}, frames),
               execute_tool("unmet_demand", {"airport": "AAA"}, frames)]
    text_ = "\n".join(key_facts(records))
    for snake in ("nas_share", "delay_rate", "ops_per_runway", "natl_rank", "_minus_", " LF"):
        assert snake not in text_
    assert "Rank change vs the balanced preset:" in text_


def test_rank_change_line_shows_moves_with_arrows(frames):
    from ui.key_facts import facts_rank
    record = execute_tool("rank_airports", {"states": ["MA", "CT"], "preset": "growth-led"}, frames)
    record.envelope["data"]["rank_change_vs_balanced"] = ["BDL: 2 -> 3 (down)", "PWM: 3 -> 2 (up)", "BOS: 1 -> 1 (same)"]
    line = next(f for f in facts_rank(record) if f.startswith("Rank change"))
    assert line == "Rank change vs the balanced preset: BDL: 2 → 3 (down); PWM: 3 → 2 (up)"


def test_unranked_list_is_capped(frames):
    from ui.key_facts import facts_rank
    record = execute_tool("rank_airports", {"states": ["MA", "CT"]}, frames)
    record.envelope["data"]["ranking"]["outside_ranking_universe"] = [
        {"airport": f"X{i:02d}", "passengers": 60_000 + i} for i in range(8)]
    line = next(f for f in facts_rank(record) if f.startswith("Shown but not ranked"))
    assert line.count("passengers)") == 5 and line.endswith("and 3 more")


# ---------------------------------------------------------------- C12 / D16 live status

TODAY = date(2026, 11, 18)
SFO_TZ = "America/Los_Angeles"


def test_live_events_are_readable_with_utc_and_local_times():
    events = [
        {"type": "groundStop", "impactingCondition": "wind", "programExpirationTime": "2026-11-18T20:00:00Z"},
        {"type": "groundDelay", "impactingCondition": "wind", "avgDelay": 92.0, "maxDelay": 341,
         "endTime": "2026-11-19T06:59:00Z"},
        {"type": "departureDelay", "reason": "WX:Thunderstorms",
         "arrivalDeparture": {"min": "16 minutes", "max": "30 minutes"}, "trend": "Increasing"},
        {"type": "airportConfig"},                                               # nothing to show
    ]
    lines = describe_events(events, SFO_TZ, TODAY)
    assert lines == [
        "Ground stop until 20:00 UTC (12:00 PST); reason: wind",
        "Ground delay program: avg 92 min, max 341 min; until Nov 19 06:59 UTC (22:59 PST); reason: wind",
        "Departure delays: 16 min to 30 min; trend increasing; Weather: thunderstorms",
    ]


def test_reason_codes_and_times():
    assert readable_reason("WX:Wind") == "Weather: wind" and readable_reason("VOL:Volume") == "Volume: volume"
    assert readable_time("2026-11-18T20:00:00Z", "America/New_York", TODAY) == "20:00 UTC (15:00 EST)"
    assert describe_event({"type": "airportConfig", "arrivalRate": 0}, None) is None


def test_live_headline_summarizes_events(frames):
    from agent.headlines import answer_headline
    from agent.tools import live_airport_status

    def feed():
        return [{"airportId": "AAA", "groundStop": {"impactingCondition": "wind",
                                                    "programExpirationTime": "2026-11-18T20:00:00Z"},
                 "groundDelay": {"avgDelay": 92.0, "maxDelay": 341, "impactingCondition": "wind"}}], "stamp"
    override = {"live_airport_status": lambda f, airport: live_airport_status(f, airport, fetch=feed, today=TODAY)}
    record = execute_tool("live_airport_status", {"airport": "AAA"}, frames, override)
    headline = answer_headline([record])
    assert "Ground stop until 20:00 UTC (15:00 EST); reason: wind; Ground delay program: avg 92 min, max 341 min" \
        in headline                                                          # AAA is in MA -> Eastern time
    assert key_facts([record])[1].startswith("Ground stop until")


# ---------------------------------------------------------------- E18 streaming in the client

class _Chunk:
    def __init__(self, content=None, usage=None):
        delta = type("D", (), {"content": content, "tool_calls": None})()
        self.choices = [type("Ch", (), {"delta": delta})()] if content is not None else []
        self.usage = usage


class _StreamStub:
    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), []
        outer = self

        class _Raw:
            def create(self, **kwargs):
                outer.calls.append(dict(kwargs))
                item = outer.outcomes.pop(0)
                if isinstance(item, Exception):
                    raise item
                return type("R", (), {"headers": {}, "parse": lambda self: iter(item)})()
        self.chat = type("Chat", (), {"completions": type("Comp", (), {"with_raw_response": _Raw()})()})()


def _gemini(stub):
    from agent.llm import DEFAULT_GEMINI_MODEL, GEMINI_BASE_URL
    return make_provider("gemini", DEFAULT_GEMINI_MODEL, GEMINI_BASE_URL, effort="minimal", client=stub,
                         sleep=lambda s: None)


def test_streamed_answer_calls_on_token_per_piece():
    usage = type("U", (), {"total_tokens": 42, "prompt_tokens": 30, "completion_tokens": 12})()
    stub = _StreamStub([[_Chunk("- one "), _Chunk("- two"), _Chunk(usage=usage)]])
    pieces = []
    resp = OpenAIChat(_gemini(stub), sleep=lambda s: None).complete(
        [{"role": "user", "content": "hi"}], [{"type": "function"}], "none", on_token=pieces.append)
    assert pieces == ["- one ", "- two"] and resp.content == "- one - two" and resp.total_tokens == 42
    assert stub.calls[0]["stream"] is True and stub.calls[0]["stream_options"] == {"include_usage": True}


def test_rejected_stream_options_are_dropped():
    class Http400(Exception):
        status_code = 400
    stub = _StreamStub([Http400("Unknown name \"stream_options\""), [_Chunk("ok")]])
    provider = _gemini(stub)
    resp = OpenAIChat(provider, sleep=lambda s: None).complete([{"role": "user", "content": "hi"}], [], "none",
                                                              on_token=lambda p: None)
    assert resp.content == "ok" and "stream_options" not in stub.calls[1] and provider.stream_usage is False


def test_cut_off_explanation_drops_the_unfinished_bullet(frames):
    from agent.llm import LLMResponse
    cut = LLMResponse("- The threshold is 3,000 mi.\n- Seat share is higher because (percent", [], 50,
                      finish_reason="length")
    result = Agent(FakeLLM([call("long_haul_share", {"airport": "AAA"}), cut]), frames).ask("long haul at AAA")
    assert result.explanation == "- The threshold is 3,000 mi."
    complete = LLMResponse("- one.\n- two ops/runway/yr", [], 50)                 # finished normally: kept
    result = Agent(FakeLLM([call("long_haul_share", {"airport": "AAA"}), complete]), frames).ask("x")
    assert result.explanation.endswith("two ops/runway/yr")


def test_stream_reports_finish_reason():
    class _Done(_Chunk):
        def __init__(self, content, reason):
            super().__init__(content)
            self.choices[0].finish_reason = reason
    stub = _StreamStub([[_Chunk("- one"), _Done(" (perc", "length")]])
    resp = OpenAIChat(_gemini(stub), sleep=lambda s: None).complete([{"role": "user", "content": "hi"}], [], "none",
                                                                   on_token=lambda p: None)
    assert resp.finish_reason == "length" and stub.calls[0]["max_tokens"] == 2000


# ---------------------------------------------------------------- Gemini first-token timeout

class _SlowStub:
    """create() waits `delays[i]` seconds before the response; a stream yields its chunks with `gap` seconds
    between them."""

    def __init__(self, delays, chunks=("ok",), gap=0.0):
        import time
        self.delays, self.calls, outer = list(delays), 0, self

        class _Raw:
            def create(self, **kwargs):
                outer.calls += 1
                time.sleep(outer.delays.pop(0))

                def stream():
                    for i, piece in enumerate(chunks):
                        if i:
                            time.sleep(gap)
                        yield _Chunk(piece)
                return type("R", (), {"headers": {}, "parse": lambda self: stream()})()
        self.chat = type("Chat", (), {"completions": type("Comp", (), {"with_raw_response": _Raw()})()})()


def _timed_gemini(stub, limit=0.2):
    from agent.llm import DEFAULT_GEMINI_MODEL, GEMINI_BASE_URL
    return make_provider("gemini", DEFAULT_GEMINI_MODEL, GEMINI_BASE_URL, client=stub, first_token_timeout=limit)


def test_no_first_token_in_time_retries_then_uses_groq():
    from tests.test_llm import _raw, groq, _Seq
    slow, sleeps = _SlowStub([1.0, 1.0]), []
    q = _Seq([_raw("from groq")])
    client = OpenAIChat(_timed_gemini(slow), groq(q), sleep=sleeps.append)
    client.start_turn()
    resp = client.complete([{"role": "user", "content": "hi"}], [])
    assert resp.model == "openai/gpt-oss-120b" and slow.calls == 2 and sleeps == [2.0]
    assert client.switch_reason == "gemini timeout (no first token within 0.2 s), after 1 retry"


def test_slow_first_attempt_then_fast_retry_stays_on_gemini():
    slow = _SlowStub([1.0, 0.0])
    client = OpenAIChat(_timed_gemini(slow), sleep=lambda s: None)
    pieces = []
    resp = client.complete([{"role": "user", "content": "hi"}], [], "none", on_token=pieces.append)
    assert resp.content == "ok" and slow.calls == 2


def test_long_stream_after_the_first_token_is_never_cut():
    """First token fast, then gaps and a total far beyond the first-token limit: the answer completes."""
    slow = _SlowStub([0.0], chunks=("- one", " two", " three."), gap=0.3)
    pieces = []
    resp = OpenAIChat(_timed_gemini(slow, limit=0.2), sleep=lambda s: None).complete(
        [{"role": "user", "content": "hi"}], [], "none", on_token=pieces.append)
    assert resp.content == "- one two three." and pieces == ["- one", " two", " three."] and slow.calls == 1


def test_gemini_provider_has_the_8_second_first_token_limit():
    from agent.llm import gemini_provider, groq_provider
    assert gemini_provider("key").first_token_timeout == 8.0
    assert groq_provider("key").first_token_timeout is None


# ---------------------------------------------------------------- F20 voice vocabulary

def test_whisper_prompt_has_domain_vocabulary():
    from agent.speech import PROMPT
    for term in ("unmet demand", "load factor", "congestion", "New England", "terminal expansion", "SNA", "BDL"):
        assert term in PROMPT


# ---------------------------------------------------------------- polish: refusals, follow-ups, live codes

def test_refusal_rule_asks_for_reason_and_in_scope_offer():
    from agent.prompts import SYSTEM_PROMPT
    rule = next(line for line in SYSTEM_PROMPT.splitlines() if line.startswith("3. Out of scope"))
    assert "max 2 sentences" in SYSTEM_PROMPT and "in-scope offer" in SYSTEM_PROMPT
    assert rule.startswith("3. Out of scope: non-US airports, construction costs")


def test_follow_ups_never_repeat_an_asked_question(frames):
    from ui.followups import suggestions
    rank = execute_tool("rank_airports", {"states": ["MA", "CT"]}, frames)
    top = rank.envelope["data"]["ranking"]["ranking"][0]["airport"]
    asked = ["Which airports rank highest?", f"show the profile of {top}"]     # case / punctuation ignored
    questions = [s.question for s in suggestions([rank], asked)]
    assert f"Show the profile of {top}" not in questions and len(questions) == 3     # a backup fills the slot
    assert questions[-1].startswith(f"Compare {top} and ")


def test_app_does_not_suggest_the_question_just_asked(app):
    from tests.test_app import ScriptedLLM
    at, _ = app(ScriptedLLM("- CCC leads."))
    at.chat_input[0].set_value("Show the profile of CCC").run()       # the fake LLM ranks MA/CT (top: CCC)
    labels = [b.label for b in at.button if b.key and "_followup_" in b.key]
    assert labels and "Show the profile of CCC" not in labels


def test_live_config_drops_raw_feed_codes():
    from data.sources.nas_status import clean_config
    assert clean_config("28L/28R (.308)") == "28L/28R"
    assert clean_config("19L INTRAIL (.25), 28R") == "19L INTRAIL, 28R"
    line = describe_event({"type": "airportConfig", "arrivalRunwayConfig": "28L, 28R (.308)",
                           "departureRunwayConfig": "1L, 1R", "arrivalRate": 36}, None)
    assert line == "Airport configuration: arrival runways 28L, 28R, departure runways 1L, 1R, arrival rate 36/hour"


class _Result:
    """Minimal TurnResult stand-in for the eval's refusal grading."""

    def __init__(self, text, explanation=None, untraceable=(), records=()):
        self.text, self.explanation = text, explanation or text
        self.untraceable_numbers, self.tool_calls = list(untraceable), list(records)
        self.error, self.grounding_issues = None, []


def test_eval_refusal_grading():
    from scripts.eval_agent import grade_refusal
    good = _Result("Construction costs are out of scope because the data covers traffic, not finance. BOS is in "
                   "scope - I can show its profile or compare it with other US airports.")
    assert grade_refusal(good, ("BOS", "Boston"))[0]
    assert not grade_refusal(_Result("Out of scope: construction costs."), ("BOS",))[0]          # no offer
    assert not grade_refusal(_Result("BOS will cost $2 billion. I can show its profile."), ("BOS",))[0]
    invented = _Result(good.text, untraceable=[{"value": "$2 billion", "context": ""}])
    assert grade_refusal(invented, ("BOS",)) == (False, "invented numbers: $2 billion")


# ---------------------------------------------------------------- display: verification note, scope wording

def test_corrected_draft_is_noted_in_the_done_line_not_the_answer(app):
    from ui.progress import done_line
    at, _ = app(_Client(["- Used the unmet_demand tool.", "- CCC leads."]))      # first draft fails, retry passes
    at.button(key="card_0").click().run()
    message = at.session_state.messages[-1]
    assert message["retried"] and message["meta"].endswith(" · auto-verified (1 correction)")
    assert not [c for c in at.caption if "first draft" in c.value]
    clean = type("R", (), {"tool_calls": [], "models": [], "total_tokens": 0, "retried": False,
                           "grounding_issues": [], "fallback_reason": None})()
    assert "auto-verified" not in done_line(clean, 1, "m")


def test_out_of_scope_key_fact_wording(frames):
    record = execute_tool("unmet_demand", {"airport": "LHR"}, frames)
    assert key_facts([record]) == [f"Out of scope: {record.error}"]
    other = execute_tool("unmet_demand", {"airport": "Atlantis"}, frames)
    assert key_facts([other])[0].startswith("Tool unmet_demand returned an error")


def test_why_label_only_when_there_are_bullets():
    from ui.render import why_markdown
    refusal = {"headline": "LHR London Heathrow Airport is outside the US scope.",
               "explanation": "Heathrow is out of scope. JFK is in scope - I can show its profile."}
    assert "Why / drivers" not in why_markdown(refusal)
    bullets = {"headline": "Top 3 ...", "explanation": "- CCC leads.\n- AAA second."}
    assert why_markdown(bullets).startswith("**Why / drivers**\n- CCC leads.")
