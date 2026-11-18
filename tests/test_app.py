"""Smoke test of app.py with streamlit.testing AppTest: fixture data + scripted fake LLM (no network)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from agent.llm import LLMResponse, ToolCall
from ui import setup

APP = str(Path(__file__).resolve().parents[1] / "app.py")


class ScriptedLLM:
    """Always: call rank_airports for MA/CT, then answer with `answer`."""

    model = "fake-model"
    reasoning_effort = None

    def __init__(self, answer: str):
        self.answer = answer
        self.round = 0

    def set_sleep(self, sleep):
        pass

    def complete(self, messages, tools, tool_choice="auto"):
        self.round += 1
        if messages[-1]["role"] == "user":
            return LLMResponse("", [ToolCall(f"c{self.round}", "rank_airports", '{"states": ["MA", "CT"]}')], 100)
        return LLMResponse(self.answer, [], 50)


@pytest.fixture
def app(monkeypatch, frames):
    def configure(api_key="test-key", answer="AAA ranks first."):
        monkeypatch.setattr(setup, "get_api_key", lambda: api_key)
        monkeypatch.setattr(setup, "get_frames", lambda: frames)
        monkeypatch.setattr(setup, "data_info", lambda: {"sources": [], "built": "2026-11-18"})
        monkeypatch.setattr(setup, "make_client", lambda: ScriptedLLM(answer))
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


def test_example_question_renders_answer_key_facts_and_evidence(app):
    at = app()
    at.sidebar.button(key="example_0").click().run()
    assert not at.exception
    text = markdown_text(at)
    assert "Which airports in New England" in text and "AAA ranks first." in text
    assert "Key facts" in text and "investment score" in text
    assert any(e.label.startswith("Evidence (1 tool call)") for e in at.expander)
    assert not at.warning


def test_typed_question_and_number_check_warning(app):
    at = app(answer="AAA scores 12,345 points.")
    at.chat_input[0].set_value("Rank Massachusetts and Connecticut airports").run()
    assert not at.exception
    assert at.warning and "12,345" in at.warning[0].value


def test_new_conversation_clears_history(app):
    at = app()
    at.sidebar.button(key="example_0").click().run()
    assert at.session_state.messages
    at.sidebar.button(key="new_conversation").click().run()
    assert at.session_state.messages == []
