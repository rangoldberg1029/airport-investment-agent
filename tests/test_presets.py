"""Named weight presets: deterministic re-rankings, labels in the headline and Key facts, follow-up mapping."""

import pytest

from agent.headlines import answer_headline
from agent.tools import TOOL_SCHEMAS, execute_tool
from scoring import ranking
from scoring.config import DEFAULT_PRESET, INVESTMENT_WEIGHTS, WEIGHT_PRESETS
from ui.followups import suggestions
from ui.key_facts import key_facts

GROWTH_QUESTION = "Rank them again with growth weighted higher"


def test_five_named_presets_each_sum_to_one():
    assert list(WEIGHT_PRESETS) == ["balanced", "demand-led", "congestion-led", "growth-led", "scale-led"]
    assert WEIGHT_PRESETS[DEFAULT_PRESET] == INVESTMENT_WEIGHTS
    for weights in WEIGHT_PRESETS.values():
        assert set(weights) == set(INVESTMENT_WEIGHTS) and sum(weights.values()) == pytest.approx(1)


@pytest.mark.parametrize("preset", list(WEIGHT_PRESETS))
def test_preset_uses_exactly_its_weights(frames, preset):
    data = execute_tool("rank_airports", {"states": ["MA", "CT"], "preset": preset}, frames).envelope["data"]
    assert data["ranking"]["weights"] == WEIGHT_PRESETS[preset] and data["ranking"]["preset"] == preset


def test_same_question_same_ranking(frames):
    runs = [execute_tool("rank_airports", {"region": "new england", "preset": "growth-led"}, frames)
            for _ in range(2)]
    assert runs[0].envelope["data"]["ranking"]["ranking"] == runs[1].envelope["data"]["ranking"]["ranking"]
    assert runs[0].llm_view == runs[1].llm_view


def test_explicit_numbers_are_custom_weights_and_unknown_preset_is_an_error(frames):
    custom = execute_tool("rank_airports", {"states": ["MA"], "weights": {"growth": 0.6}}, frames)
    assert custom.envelope["data"]["ranking"]["preset"] is None
    bad = execute_tool("rank_airports", {"states": ["MA"], "preset": "fastest"}, frames)
    assert bad.error and "unknown preset 'fastest'" in bad.error and "growth-led" in bad.error


def test_preset_name_and_weights_in_headline_and_key_facts(frames):
    record = execute_tool("rank_airports", {"states": ["MA", "CT"], "preset": "scale-led"}, frames)
    label = "preset scale-led (demand pressure 25%, growth 15%, congestion 15%, scale 45%)"
    assert f"Weights: {label}." in answer_headline([record])
    assert key_facts([record])[0] == f"Weights: {label}"


def test_rank_stability_uses_all_presets(frames):
    data = ranking.rank_stability(frames, states=["MA", "CT"])["data"]
    assert list(data["rankings"]) == list(WEIGHT_PRESETS)


# ---------------------------------------------------------------- "growth weighted higher" -> preset growth-led

def test_growth_question_maps_to_growth_led_preset(frames):
    rank = execute_tool("rank_airports", {"region": "new england"}, frames)
    suggestion = suggestions([rank])[0]
    assert suggestion.question == GROWTH_QUESTION
    assert (suggestion.tool, suggestion.args) == ("rank_airports", {"region": "new england", "preset": "growth-led"})


def test_tool_description_tells_the_llm_to_use_presets():
    schema = next(s["function"] for s in TOOL_SCHEMAS if s["function"]["name"] == "rank_airports")
    props = schema["parameters"]["properties"]
    assert props["preset"]["enum"] == list(WEIGHT_PRESETS)
    assert "'weight growth higher' -> growth-led" in props["preset"]["description"]
    assert props["weights"]["description"].startswith("ONLY if the user gives explicit numbers")


def test_eval_expects_the_growth_led_preset_for_the_growth_question():
    from scripts.eval_agent import SCRIPT, _rerank_ok
    question, tools, _, check = SCRIPT[1]
    assert question.startswith(GROWTH_QUESTION) and tools == {"rank_airports"} and check is _rerank_ok
    assert _rerank_ok({"region": "new england", "preset": "growth-led"})
    assert not _rerank_ok({"region": "new england", "weights": {"growth": 0.45}})     # no numbers were given
    assert not _rerank_ok({"region": "new england", "preset": "demand-led"})
