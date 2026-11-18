"""Agent loop with a scripted fake LLM (no network)."""

import json

from agent.agent import MAX_TEXT_TURNS, OLD_ANSWER_CHARS, Agent
from agent.llm import FRIENDLY_RATE_LIMIT, LLMResponse, RateLimited, ToolCall


class FakeLLM:
    """Replays scripted responses; records every message list it receives."""

    def __init__(self, script):
        self.script = list(script)
        self.seen: list[list[dict]] = []
        self.tool_choices: list[str] = []
        self.streamed: list[bool] = []

    def complete(self, messages, tools, tool_choice="auto", on_token=None):
        self.seen.append(json.loads(json.dumps(messages)))
        self.tool_choices.append(tool_choice)
        self.streamed.append(on_token is not None)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        stream(item, on_token)
        return item


def stream(resp, on_token) -> None:
    """Simulate a streamed answer: pass the text to on_token in word-sized pieces."""
    if on_token and resp.content:
        for piece in resp.content.split(" "):
            on_token(piece + " ")


def call(name, args, cid="c1"):
    return LLMResponse("", [ToolCall(cid, name, json.dumps(args))], total_tokens=100)


def text(t):
    return LLMResponse(t, [], total_tokens=50)


def test_tool_then_answer(frames):
    fake = FakeLLM([call("long_haul_share", {"airport": "AAA"}), text("23.08% of departures are long-haul.")])
    result = Agent(fake, frames).ask("long haul at AAA?")
    assert [r.name for r in result.tool_calls] == ["long_haul_share"]
    assert result.untraceable_numbers == [] and result.llm_calls == 2 and result.total_tokens == 150
    tool_msg = fake.seen[1][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "c1"
    assert '"departure share":"23.08%"' in tool_msg["content"]                 # readable keys for the LLM


def test_two_tools_in_one_turn(frames):
    two = LLMResponse("", [ToolCall("a", "long_haul_share", '{"airport":"AAA"}'),
                           ToolCall("b", "unmet_demand", '{"airport":"AAA"}')])
    result = Agent(FakeLLM([two, text("done")]), frames).ask("both")
    assert [r.name for r in result.tool_calls] == ["long_haul_share", "unmet_demand"]


def test_tool_error_is_returned_to_llm_and_loop_recovers(frames):
    fake = FakeLLM([call("unmet_demand", {"airport": "Atlantis"}), text("Which airport do you mean?")])
    result = Agent(fake, frames).ask("unmet demand at Atlantis")
    assert result.tool_calls[0].error and "unknown airport" in fake.seen[1][-1]["content"]
    assert result.text == "Which airport do you mean?"


def test_one_tool_round_then_streamed_explanation_without_tools(frames):
    """After the tools ran, the explanation call cannot call tools again (tool_choice none) and is streamed."""
    fake = FakeLLM([call("long_haul_share", {"airport": "AAA"}), text("final")])
    events = []
    result = Agent(fake, frames).ask("long haul at AAA", on_event=lambda e, **kw: events.append((e, kw)))
    assert fake.tool_choices == ["auto", "none"] and fake.streamed == [False, True]
    assert result.explanation == "final"
    assert result.text.startswith("**Answer** - AAA AAA Intl: 23.08% of 156 passenger departures")
    names = [e for e, _ in events]
    assert names.index("facts_ready") < names.index("token")                  # facts shown before the bullets
    facts = next(kw for e, kw in events if e == "facts_ready")
    assert facts["headline"].startswith("AAA AAA Intl") and facts["records"][0].name == "long_haul_share"
    assert "".join(kw["text"] for e, kw in events if e == "token").strip() == "final"


def test_memory_follow_up_sees_previous_turn_as_text_and_tool_summary(frames):
    fake = FakeLLM([call("unmet_demand", {"airport": "AAA"}), text("AAA answer"),
                    call("unmet_demand", {"airport": "BBB"}, "c2"), text("BBB answer")])
    agent = Agent(fake, frames)
    agent.ask("unmet demand at AAA?")
    agent.ask("what about BBB?")
    second_turn_first_call = fake.seen[2]
    contents = [m.get("content") or "" for m in second_turn_first_call]
    assert "unmet demand at AAA?" in contents and any(c.endswith("AAA answer") for c in contents)
    assert any(c.startswith('[Tools used: unmet_demand({"airport": "AAA"})]') for c in contents)
    assert not any(m["role"] == "tool" for m in second_turn_first_call)    # earlier tool outputs: summary only


def test_trimming_drops_old_tool_messages_and_keeps_pairs_valid(frames):
    script = []
    for i in range(MAX_TEXT_TURNS + 2):
        script += [call("long_haul_share", {"airport": "AAA"}, f"id{i}"), text(f"answer {i}")]
    fake = FakeLLM(script)
    agent = Agent(fake, frames)
    for i in range(MAX_TEXT_TURNS + 2):
        agent.ask(f"question {i}")
    last = fake.seen[-1]
    users = [m["content"] for m in last if m["role"] == "user"]
    assert "question 0" not in users and len(users) == MAX_TEXT_TURNS + 1     # 6 history + current
    tool_ids = {m["tool_call_id"] for m in last if m["role"] == "tool"}
    call_ids = {c["id"] for m in last if m.get("tool_calls") for c in m["tool_calls"]}
    assert tool_ids == call_ids == {f"id{MAX_TEXT_TURNS + 1}"}                 # current turn's tool output only


def test_long_conversations_do_not_grow_with_tool_outputs(frames):
    """Turn 5's prompt holds earlier turns only as text + one-line tool summaries."""
    script = []
    for i in range(5):
        script += [call("rank_airports", {"states": ["MA", "CT"]}, f"id{i}"), text(f"- answer {i}")]
    fake = FakeLLM(script)
    agent = Agent(fake, frames)
    for i in range(5):
        agent.ask(f"rank question {i}")
    size = [len(json.dumps(fake.seen[2 * i + 1])) for i in range(5)]       # explanation call of each turn
    tool_view = len(agent.turns[0].records[0].llm_view)
    assert size[4] - size[0] < tool_view                                     # growth << one extra tool output


def test_rate_limited_turn_returns_friendly_message(frames):
    result = Agent(FakeLLM([RateLimited(FRIENDLY_RATE_LIMIT)]), frames).ask("hi")
    assert result.error == "rate_limited" and "rate limit was reached" in result.text


def test_provider_error_does_not_crash(frames):
    result = Agent(FakeLLM([RuntimeError("boom")]), frames).ask("hi")
    assert result.error == "boom" and "failed" in result.text


def test_number_check_flags_invented_numbers(frames):
    fake = FakeLLM([call("long_haul_share", {"airport": "AAA"}), text("Share is 23.08%, up from 19.4% last year.")])
    result = Agent(fake, frames).ask("long haul?")
    assert [n["value"] for n in result.untraceable_numbers] == ["19.4%"]


class QuotingLLM:
    """First call: request compare_congestion. Second call: quote numbers that exist ONLY in the
    compact LLM view (precomputed difference + net contribution difference)."""

    def __init__(self):
        self.calls = 0

    def complete(self, messages, tools, tool_choice="auto", on_token=None):
        self.calls += 1
        if self.calls == 1:
            return call("compare_congestion", {"airports": ["AAA", "BBB"]})
        view = json.loads(messages[-1]["content"])["data"]
        delay_diff = view["differences"]["AAA vs BBB"]["arrival delay rate"]
        net = view["drivers of index difference"]["AAA vs BBB"]["net difference"]
        return text(f"AAA vs BBB: arrival delay rate {delay_diff}, net contribution difference {net}.")


def test_number_check_accepts_precomputed_differences_from_llm_view(frames):
    result = Agent(QuotingLLM(), frames).ask("compare AAA and BBB")
    assert "net contribution difference" in result.text
    assert result.untraceable_numbers == []


def test_number_check_uses_earlier_turns(frames):
    fake = FakeLLM([call("long_haul_share", {"airport": "AAA"}), text("23.08%"), text("As said, 23.08%.")])
    agent = Agent(fake, frames)
    agent.ask("long haul?")
    assert agent.ask("repeat?").untraceable_numbers == []


def test_older_answers_truncated_latest_kept_in_full(frames):
    long_answer = "x" * (OLD_ANSWER_CHARS + 200)
    fake = FakeLLM([text("A " + long_answer), text("B " + long_answer), text("C")])
    agent = Agent(fake, frames)
    agent.ask("first")
    agent.ask("second")
    agent.ask("third")
    assistant = [m["content"] for m in fake.seen[-1] if m["role"] == "assistant"]
    assert assistant[0].startswith("A ") and assistant[0].endswith("[...]") and len(assistant[0]) < OLD_ANSWER_CHARS + 10
    assert assistant[1] == "B " + long_answer                               # latest completed turn: full text
