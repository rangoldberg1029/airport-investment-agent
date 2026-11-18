"""Grounding check + follow-up context + retry, incl. the exact bug found in manual testing."""

import json

from agent.agent import Agent
from agent.guardrails import airport_codes, grounding_issues
from agent.llm import LLMResponse
from agent.tools import TOOLS
from tests.test_agent import FakeLLM, call, text

KNOWN = {"SFO", "DFW", "DEN", "BOS", "LAX", "FAA", "NAS", "AND"}
TOOL_NAMES = list(TOOLS)


def issues(answer, user="Why is it high?", called=(), grounded=""):
    return grounding_issues(answer, user, list(called), TOOL_NAMES, grounded, KNOWN)


# ---------------------------------------------------------------- rules

def test_rule_a_airport_codes_must_be_grounded():
    assert airport_codes("SFO and DFW, per FAA and NAS data", KNOWN) == {"SFO", "DFW"}   # acronyms ignored
    found = issues("SFO falls to rank 3; DFW moves ahead.", grounded='{"airport": "SFO"}')
    assert found == ["airport codes not found in any tool output or question: DFW"]
    assert issues("BOS leads.", grounded="... BOS ...") == []


def test_rule_b_answer_must_not_claim_uncalled_tools():
    assert issues("I used the rank_airports tool.")[0].startswith("answer names tool(s) not called")
    assert issues("Based on the rank airports tool ...")                       # spaced variant
    assert issues("I used the rank_airports tool.", called=["rank_airports"]) == []


def test_rule_c_rank_compare_weights_need_a_tool_call():
    for request in ("Rank them again with growth weighted higher", "Compare SFO and LAX",
                    "Re-rank with different weights", "Calculate the score again"):
        assert issues("Done.", user=request) == ["the request asks to rank/compare/compute or change weights, "
                                                 "but no tool was called"], request
    assert issues("Done.", user="Rank them again", called=["rank_airports"]) == []
    assert issues("SNA has the higher NAS share.", user="Why is SNA lower?", grounded="SNA") == []


# ---------------------------------------------------------------- regression: the manual-testing bug

INVENTED = ("Re-ranking all US large hubs with growth weighted higher (default growth weight 0.20): "
            "SFO falls to rank 3, while DFW and DEN move ahead. Used the rank_airports tool.")


def test_invented_rerank_triggers_retry_with_required_tool_call(frames):
    for code, city in [("DFW", "Dallas"), ("DEN", "Denver")]:          # make the invented codes 'known' airports
        frames.airports.loc[len(frames.airports)] = {**frames.airports.iloc[0].to_dict(), "iata": code, "city": city}
    fake = FakeLLM([
        call("rank_airports", {"region": "new england", "top_n": 10}, "t1"),   # turn 1: correct tool call
        text("AAA leads New England."),
        text(INVENTED),                                                         # turn 2, draft 1: no tool, invented
        call("rank_airports", {"region": "new england", "top_n": 10, "preset": "growth-led"}, "t2"),
        text("With growth weighted higher, CCC ranks first in New England."),
    ])
    agent = Agent(fake, frames)
    agent.ask("Which airports in New England are strong candidates for terminal expansion?")
    result = agent.ask("Rank them again with growth weighted higher")

    # follow-up context was injected into turn 2 (structured: last tool call name + arguments)
    turn2_first_user_msg = fake.seen[2][-1]["content"]
    assert 'previous tool call in this conversation: rank_airports({"region": "new england"' in turn2_first_user_msg
    # retry forced a tool call
    assert fake.tool_choices[2] == "auto" and fake.tool_choices[3] == "required"
    assert "rejected by an automatic check" in fake.seen[3][-1]["content"]
    # final result: grounded, tool called with the same region and the growth-led preset
    assert result.retried and result.grounding_issues == []
    assert [r.name for r in result.tool_calls] == ["rank_airports"]
    assert result.tool_calls[0].args["region"] == "new england" and result.tool_calls[0].args["preset"] == "growth-led"
    assert result.explanation.startswith("With growth weighted higher")
    assert result.text.startswith("**Answer** - Top ") and "preset growth-led" in result.headline
    # the invented draft never enters the conversation history
    assert agent.turns[0].answer.endswith("AAA leads New England.") and agent.turns[1].answer == result.text
    assert all(INVENTED not in json.dumps(m) for m in fake.seen[-1])


def test_still_failing_after_retry_is_reported(frames):
    fake = FakeLLM([text("SFO is first. Used the rank_airports tool."),
                    text("SFO is first. Used the rank_airports tool.")])     # ignores tool_choice=required
    result = Agent(fake, frames).ask("Rank airports by growth")
    assert result.retried and len(result.grounding_issues) == 3
    assert fake.tool_choices == ["auto", "required"]


def test_explaining_previous_output_without_tool_is_allowed(frames):
    fake = FakeLLM([call("long_haul_share", {"airport": "AAA"}), text("AAA: 23.08% long-haul."),
                    text("Because AAA has routes at 3,000 and 4,500 mi.")])
    agent = Agent(fake, frames)
    agent.ask("Long-haul share at AAA?")
    result = agent.ask("Why is AAA so high?")
    assert not result.retried and result.grounding_issues == [] and len(fake.seen) == 3


class _NoToolLLM:
    """Never calls tools; used to test the follow-up context content only."""

    def complete(self, messages, tools, tool_choice="auto"):
        return LLMResponse("ok", [], 1)


def test_no_context_before_first_tool_call(frames):
    agent = Agent(_NoToolLLM(), frames)
    assert agent.followup_context() == ""
    agent.ask("hello")
    assert agent.followup_context() == ""
