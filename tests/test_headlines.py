"""Deterministic answer headlines, combining them with the LLM's bullets, and the direct (fast) path."""

import pytest

from agent.agent import Agent  # noqa: F401  (used below)
from agent.headlines import answer_headline, compose, why_bullets
from agent.llm import QUOTA_MESSAGE, QuotaExhausted
from agent.tools import execute_tool, live_airport_status
from tests.test_agent import FakeLLM, call, text


# ---------------------------------------------------------------- headlines

def test_rank_headline_is_built_from_the_ranking(frames):
    record = execute_tool("rank_airports", {"states": ["MA", "CT"]}, frames)
    rows = record.envelope["data"]["ranking"]["ranking"][:3]
    items = "; ".join(f"{r['rank']}. {r['airport']} {r['airport']} Intl ({r['investment_score']})" for r in rows)
    assert answer_headline([record]) == (
        f"Top {len(rows)} candidates in MA/CT for terminal expansion (investment score 0-100, "
        f"Sep 2025 - Aug 2026): {items}. Preset balanced (default): demand pressure 35%, growth 25%, "
        f"congestion 25%, scale 15%.")


def test_rank_headline_names_the_preset_and_its_weights(frames):
    headline = answer_headline([execute_tool("rank_airports", {"region": "new england", "preset": "growth-led"},
                                             frames)])
    assert headline.startswith("Top ") and "New England candidates" in headline
    assert "Preset growth-led: demand pressure 25%, growth 45%, congestion 15%, scale 15%." in headline
    custom = answer_headline([execute_tool("rank_airports", {"region": "new england", "weights": {"growth": 0.5}},
                                           frames)])
    assert "Custom weights: " in custom


@pytest.mark.parametrize("name,args,expected", [
    ("compare_congestion", {"airports": ["AAA", "BBB"]}, "Congestion index (0-100, higher = more congested"),
    ("long_haul_share", {"airport": "AAA"}, "AAA AAA Intl: 23.08% of 156 passenger departures fly 3,000 mi or more"),
    ("unmet_demand", {"airport": "AAA"}, "AAA AAA Intl (large hub): unmet-demand proxy score"),
    ("get_airport_profile", {"airport": "AAA"}, "AAA AAA Intl (large hub, MA, Sep 2025 - Aug 2026):"),
    ("explain_methodology", {"topic": "investment_score"}, "Investment score (0-100) = weighted national"),
])
def test_every_tool_has_a_headline(frames, name, args, expected):
    assert answer_headline([execute_tool(name, args, frames)]).startswith(expected)


def test_compare_headline_states_the_largest_driver(frames):
    headline = answer_headline([execute_tool("compare_congestion", {"airports": ["AAA", "BBB"]}, frames)])
    assert "Largest single driver of the difference: arrival delay rate raises AAA by 11.67 pts." in headline


# ---------------------------------------------------------------- "why is X lower?" gets its own headline

def test_why_lower_headline_explains_from_the_drivers(frames):
    record = execute_tool("compare_congestion", {"airports": ["AAA", "BBB"]}, frames)
    rows = {r["airport"]: r["congestion_index"] for r in record.envelope["data"]["congestion"]["airports"]}
    low, high = sorted(rows, key=rows.get)
    headline = answer_headline([record], f"Why is {low} lower?")
    assert headline.startswith(f"Why {low}'s congestion index is lower ({low} {rows[low]} vs {high} {rows[high]}):")
    assert f"raises {high} by" in headline and "Net:" in headline
    assert not headline.startswith("Congestion index (0-100")                  # not the comparison headline again


def test_why_question_with_a_wrong_premise_is_corrected(frames):
    record = execute_tool("compare_congestion", {"airports": ["AAA", "BBB"]}, frames)
    rows = {r["airport"]: r["congestion_index"] for r in record.envelope["data"]["congestion"]["airports"]}
    high = max(rows, key=rows.get)
    assert answer_headline([record], f"Why is {high} lower?").startswith(f"{high} is not lower: its congestion index "
                                                                          "is higher")


def test_out_of_scope_airport_headline(frames):
    record = execute_tool("long_haul_share", {"airport": "LHR"}, frames)
    assert answer_headline([record]).startswith("LHR London Heathrow Airport is outside the US scope")


def test_national_ranking_note_without_previous_context(frames):
    fake = FakeLLM([call("rank_airports", {}), text("- answer")])
    result = Agent(fake, frames).ask("Which is third now?")
    assert "No previous ranking in this conversation; showing the national ranking." in result.headline
    assert "no previous result in this conversation" in fake.seen[0][-1]["content"]      # clarify-note for the LLM


def test_unmet_headline_has_capacity_gap_and_proxy_caveat(frames):
    headline = answer_headline([execute_tool("unmet_demand", {"airport": "AAA"}, frames)])
    assert "Capacity gap at target load factor 0.85" in headline and "true spilled demand is not observable" in headline


def test_no_headline_for_errors(frames):
    assert answer_headline([execute_tool("unmet_demand", {"airport": "Atlantis"}, frames)]) == ""
    assert answer_headline([]) == ""


def test_live_headline_says_context_only(frames):
    def fetch():
        return [], "2026-11-18T10:00:00Z"
    no_network = {"live_airport_status": lambda f, airport: live_airport_status(f, airport, fetch=fetch)}
    record = execute_tool("live_airport_status", {"airport": "AAA"}, frames, no_network)
    assert answer_headline([record]).endswith("Live context only; not used in any score.")


# ---------------------------------------------------------------- combining with the LLM text

def test_why_bullets_strips_answer_and_heading_and_caps_at_four():
    llm = ("**Answer** - BOS ranks first.\n\n**Why / drivers**\n- one\n- two\n- three\n- four\n- five")
    assert why_bullets(llm) == "- one\n- two\n- three\n- four"


def test_why_bullets_keeps_plain_bullets():
    assert why_bullets("- a\n- b") == "- a\n- b"
    assert why_bullets("* a\n1. b") == "* a\n1. b"


def test_compose():
    assert compose("H", "- a") == "**Answer** - H\n\n**Why / drivers**\n- a"
    assert compose("", "plain reply") == "plain reply"
    assert compose("H", "") == "**Answer** - H"
    assert compose("H", "error", why_heading=False) == "**Answer** - H\n\nerror"


# ---------------------------------------------------------------- agent: checks run on the LLM part only

def test_headline_numbers_are_not_checked_but_llm_numbers_are(frames):
    fake = FakeLLM([call("long_haul_share", {"airport": "AAA"}), text("- Share is 23.08%, up from 19.4%.")])
    result = Agent(fake, frames).ask("long-haul share at AAA?")
    assert result.headline.startswith("AAA AAA Intl: 23.08%")
    assert [n["value"] for n in result.untraceable_numbers] == ["19.4%"]


# ---------------------------------------------------------------- direct (fast) path

def test_direct_path_runs_the_tool_then_one_llm_call(frames):
    fake = FakeLLM([text("- AAA has the highest load factor.")])
    agent = Agent(fake, frames)
    events = []
    result = agent.ask_direct("Rank MA and CT", "rank_airports", {"states": ["MA", "CT"]},
                              on_event=lambda e, **kw: events.append(e))
    assert fake.tool_choices == ["none"] and result.llm_calls == 1
    assert events[:2] == ["tool_start", "tool_done"] and "llm_call" in events
    assert result.text.startswith("**Answer** - Top ") and result.text.endswith("- AAA has the highest load factor.")
    # the LLM saw the tool call and its result as if it had chosen it
    sent = fake.seen[0]
    assert sent[-2]["tool_calls"][0]["function"]["name"] == "rank_airports" and sent[-1]["role"] == "tool"


def test_direct_call_is_recorded_for_follow_ups(frames):
    fake = FakeLLM([text("- why"), call("rank_airports", {"states": ["MA", "CT"], "weights": {"growth": 0.5}}),
                    text("- re-ranked")])
    agent = Agent(fake, frames)
    agent.ask_direct("Rank MA and CT", "rank_airports", {"states": ["MA", "CT"]})
    assert agent.last_tool_call().name == "rank_airports"
    agent.ask("Rank them again with growth weighted higher")
    assert 'previous tool call in this conversation: rank_airports({"states": ["MA", "CT"]})' in \
        fake.seen[1][-1]["content"]


def test_direct_path_keeps_the_tool_result_when_the_llm_fails(frames):
    fake = FakeLLM([QuotaExhausted(QUOTA_MESSAGE)])
    agent = Agent(fake, frames)
    result = agent.ask_direct("long-haul at AAA", "long_haul_share", {"airport": "AAA"})
    assert result.error == "quota_exhausted" and result.headline.startswith("AAA AAA Intl: 23.08%")
    assert result.text.startswith("**Answer** - AAA") and result.text.endswith(QUOTA_MESSAGE)
    assert len(agent.turns) == 1 and agent.turns[0].answer.startswith("**Answer** - AAA")


def test_direct_path_grounding_retry_stays_without_tools(frames):
    fake = FakeLLM([text("- ZZZ is first. Used the unmet_demand tool."), text("- AAA leads.")])
    agent = Agent(fake, frames)
    result = agent.ask_direct("Rank MA and CT", "rank_airports", {"states": ["MA", "CT"]})
    assert fake.tool_choices == ["none", "none"] and result.retried and not result.grounding_issues
    assert "rejected by an automatic check" in fake.seen[1][-3]["content"]
