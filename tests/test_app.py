"""Smoke test of app.py with streamlit.testing AppTest: fixture data + scripted fake LLM (no network)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from agent.llm import LLMResponse, ToolCall
from ui import setup
from ui.cache import AnswerCache

APP = str(Path(__file__).resolve().parents[1] / "app.py")


class ScriptedLLM:
    """Always: call rank_airports for MA/CT, then answer with `answer`."""

    model = "fake-model"
    reasoning_effort = None

    calls = 0                      # class-level: counts LLM calls across instances in one test

    def __init__(self, answer: str):
        self.answer = answer
        self.round = 0

    def set_sleep(self, sleep):
        pass

    def complete(self, messages, tools, tool_choice="auto"):
        self.round += 1
        ScriptedLLM.calls += 1
        if messages[-1]["role"] == "user" and tool_choice != "none":
            return LLMResponse("", [ToolCall(f"c{self.round}", "rank_airports", '{"states": ["MA", "CT"]}')], 100)
        return LLMResponse(self.answer, [], 50)


@pytest.fixture
def app(monkeypatch, frames):
    def configure(api_key="test-key", answer="AAA ranks first.", client=None):
        monkeypatch.setattr(setup, "get_api_key", lambda: api_key)
        monkeypatch.setattr(setup, "get_frames", lambda: frames)
        monkeypatch.setattr(setup, "data_info", lambda: {"sources": [], "built": "2026-11-18"})
        monkeypatch.setattr(setup, "make_client", lambda: client or ScriptedLLM(answer))
        cache = AnswerCache()
        monkeypatch.setattr(setup, "get_answer_cache", lambda: cache)
        ScriptedLLM.calls = 0
        at = AppTest.from_file(APP, default_timeout=60)
        at.run()
        return at
    return configure


def markdown_text(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_missing_key_shows_setup_message(app):
    at = app(api_key=None)
    assert not at.exception
    assert any("Setup needed" in i.value for i in at.info)
    assert at.chat_input[0].disabled


def test_examples_shown_in_exactly_one_place(app):
    at = app()
    # empty state: main-area cards only, no sidebar list
    assert sorted(b.key for b in at.button if b.key and b.key.startswith("card_")) == [f"card_{i}" for i in range(4)]
    assert not [b for b in at.sidebar.button if b.key and b.key.startswith("example_")]
    at.button(key="card_0").click().run()
    # conversation started: sidebar list only, no cards
    assert [b.key for b in at.sidebar.button if b.key and b.key.startswith("example_")] == \
        [f"example_{i}" for i in range(4)]
    assert not [b for b in at.button if b.key and b.key.startswith("card_")]
    # back to the empty state after "New conversation"
    at.sidebar.button(key="new_conversation").click().run()
    assert not [b for b in at.sidebar.button if b.key and b.key.startswith("example_")]
    assert [b for b in at.button if b.key and b.key.startswith("card_")]


def test_example_question_renders_answer_key_facts_and_evidence(app):
    at = app()
    at.button(key="card_0").click().run()
    assert not at.exception
    text = markdown_text(at)
    assert "Which airports in New England" in text and "AAA ranks first." in text
    assert "**Answer** - Top " in text and "New England candidates" in text      # computed headline
    assert "**Why / drivers**" in text
    assert "Key facts" in text and "investment score" in text
    assert ScriptedLLM.calls == 1                                                # fast path: one LLM call
    last = at.session_state.messages[-1]
    assert last["records"][0].name == "rank_airports" and last["records"][0].args == {"region": "new england"}
    assert any(e.label.startswith("Evidence (1 tool call)") for e in at.expander)
    assert not at.warning


def test_typed_question_and_number_check_warning(app):
    at = app(answer="AAA scores 12,345 points.")
    at.chat_input[0].set_value("Rank Massachusetts and Connecticut airports").run()
    assert not at.exception
    assert at.warning and "12,345" in at.warning[0].value


def test_new_conversation_clears_history(app):
    at = app()
    at.button(key="card_0").click().run()
    assert at.session_state.messages
    at.sidebar.button(key="new_conversation").click().run()
    assert at.session_state.messages == []


def test_done_line_shown_once(app):
    at = app()
    at.button(key="card_0").click().run()
    assert not [c for c in at.caption if c.value.startswith("Done")]     # only in the status label (live)


def test_first_question_cached_across_conversations(app):
    at = app()
    at.button(key="card_0").click().run()
    calls_after_first = ScriptedLLM.calls
    assert calls_after_first == 1                                         # fast path: tool ran directly

    at.sidebar.button(key="new_conversation").click().run()
    at.button(key="card_0").click().run()                                 # empty state again: card
    assert not at.exception
    assert ScriptedLLM.calls == calls_after_first                         # served from cache: no LLM call
    last = at.session_state.messages[-1]
    assert last["cached"] and last["text"].endswith("AAA ranks first.") and last["meta"].startswith("Cached result")
    assert len(at.session_state.agent.turns) == 1                         # restored for follow-ups


def test_follow_up_is_not_cached(app):
    at = app()
    at.button(key="card_0").click().run()                                 # empty state: card
    at.sidebar.button(key="example_0").click().run()                      # same text, now a follow-up (sidebar)
    assert ScriptedLLM.calls == 2 and not at.session_state.messages[-1].get("cached")   # 1 per fast-path answer


class InventingLLM:
    """Never calls a tool and names one it did not call: fails the grounding check twice."""

    model = "fake-model"
    reasoning_effort = None

    def set_sleep(self, sleep):
        pass

    def complete(self, messages, tools, tool_choice="auto"):
        return LLMResponse("SFO falls to rank 3. Used the rank_airports tool.", [], 10)


def test_grounding_failure_shows_warning_and_is_not_cached(app):
    at = app(client=InventingLLM())
    at.chat_input[0].set_value("Rank them again with growth weighted higher").run()
    assert not at.exception
    warning = " ".join(w.value for w in at.warning)
    assert "Grounding check failed" in warning and "rank_airports" in warning
    assert at.session_state.messages[-1]["retried"]
    assert len(setup.get_answer_cache()) == 0
