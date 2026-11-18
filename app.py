"""Streamlit chat UI for the Airport Investment Intelligence Agent.  Run:  streamlit run app.py"""

import hashlib
import time

import streamlit as st

from agent.agent import Agent, TurnResult
from agent.headlines import answer_headline, compose
from scoring.common import window_label
from scoring.config import WINDOW
from agent.prompts import PROMPT_VERSION
from ui import setup
from ui.cache import cache_key
from ui.progress import StatusReporter, done_line
from ui.render import (
    EXAMPLE_CALLS, EXAMPLES, about_markdown, render_assistant, render_checks, render_details, render_followups,
)

st.set_page_config(page_title="Airport Investment Intelligence", page_icon="✈️", layout="wide")

st.title("Airport Investment Intelligence")
st.markdown("Find US airports where terminal and capacity investments are most likely to pay off, "
            "based on passenger demand versus capacity.")

api_key = setup.get_api_key()
frames = setup.get_frames()

with st.expander("About this agent", expanded=False):
    st.markdown(about_markdown(frames))


def ask_later(question: str, tool: str | None = None, args: dict | None = None) -> None:
    """Button callback: queue a question (runs before the rerun, so the page redraws with it already queued).
    With a known tool + arguments (example cards, suggested follow-ups) the answer takes the fast path."""
    st.session_state.pending = {"text": question, "tool": tool, "args": args or {}}


def new_conversation() -> None:
    """Button callback: clear the conversation before the page redraws (so the empty state shows at once)."""
    st.session_state.messages = []
    if "agent" in st.session_state:
        st.session_state.agent.reset()
    st.session_state.pop("voice_text", None)


# ---------------------------------------------------------------- session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state and api_key:
    st.session_state.agent = Agent(setup.make_client(), frames)
st.session_state.setdefault("pending", None)

# Empty state = no conversation yet and nothing queued. The example questions are shown in exactly one place:
# as cards in the main area on the empty state, otherwise in the sidebar.
incoming = st.session_state.get("chat_input") or st.session_state.pending
empty_state = not st.session_state.messages and not incoming

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    if not empty_state:
        st.subheader("Example questions")
        for i, q in enumerate(EXAMPLES):
            st.button(q, key=f"example_{i}", width="stretch", disabled=not api_key, on_click=ask_later,
                      args=(q, *EXAMPLE_CALLS[i]))
    st.button("New conversation", key="new_conversation", type="primary", width="stretch",
              on_click=new_conversation)

    st.subheader("Ask by voice (beta)")
    if not api_key:
        st.caption("Available once the API key is configured.")
    elif not setup.voice_available():
        st.caption("Voice input requires GROQ_API_KEY")
    elif not hasattr(st, "audio_input"):
        st.caption("Voice input needs a newer Streamlit version.")
    else:
        audio = st.audio_input("Record a question", key="voice_audio", label_visibility="collapsed")
        if audio is not None:
            data = audio.getvalue()
            digest = hashlib.sha1(data).hexdigest()
            if st.session_state.get("voice_digest") != digest:      # transcribe each recording once
                st.session_state.voice_digest = digest
                try:
                    from agent.speech import transcribe

                    with st.spinner("Transcribing..."):
                        st.session_state.voice_text = transcribe(data)
                except Exception as exc:
                    st.session_state.voice_text = ""
                    st.warning(f"Voice transcription is unavailable right now ({type(exc).__name__}). "
                               "Please type your question.")
        if st.session_state.get("voice_text"):
            edited = st.text_area("Transcript (check airport codes, then send)", key="voice_text", height=90)
            if st.button("Send", key="voice_send", width="stretch"):
                st.session_state.pending = {"text": edited.strip()}

    st.divider()
    st.caption("**Model:** " + (setup.model_info(st.session_state.agent.client)
                                if "agent" in st.session_state else "not configured"))
    info = setup.data_info()
    st.caption(f"**Data period:** {window_label(WINDOW)} (trailing 12 months)"
               + (f"  \n**Data built:** {info['built']}" if info.get("built") else ""))

# ---------------------------------------------------------------- setup notice
if not api_key:
    st.info("**Setup needed: no LLM API key found.**\n\n"
            "1. Create a free Google AI Studio key (aistudio.google.com/apikey). Optional: a free Groq key "
            "(console.groq.com/keys) for voice input and automatic failover.\n"
            "2. Copy `.env.example` to `.env` and set `GEMINI_API_KEY=...` (required) and `GROQ_API_KEY=...` "
            "(optional). When deployed, add the same keys to Streamlit secrets "
            "(see `.streamlit/secrets.toml.example`).\n"
            "3. Restart the app: `streamlit run app.py`.\n\n"
            "The data and methodology above are available without a key.")

# ---------------------------------------------------------------- empty state
EXAMPLE_TITLES = ["Expansion candidates", "Congestion comparison", "Long-haul share", "Unmet demand"]
if empty_state:
    st.markdown("#### Welcome")
    st.markdown("Ask about passenger demand, congestion or expansion candidates at US airports, "
                "or start with one of these questions:")
    cols = st.columns(2)
    for i, (title, q) in enumerate(zip(EXAMPLE_TITLES, EXAMPLES)):
        with cols[i % 2].container(border=True):
            st.markdown(f"**{title}**")
            st.button(q, key=f"card_{i}", type="tertiary", disabled=not api_key, on_click=ask_later,
                      args=(q, *EXAMPLE_CALLS[i]))

# ---------------------------------------------------------------- history
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["text"])
        else:
            render_assistant(message, key=f"msg{idx}", on_followup=ask_later)

# ---------------------------------------------------------------- new question
typed = st.chat_input("Ask about US airport demand, congestion or expansion candidates",
                      disabled=not api_key, key="chat_input")
request = {"text": typed} if typed else st.session_state.pending
st.session_state.pending = None

RATE_LIMITED_NOTE = "The LLM provider is temporarily at capacity. Please try again in about a minute."


def error_note(result: TurnResult) -> str:
    return RATE_LIMITED_NOTE if result.error == "rate_limited" else result.explanation


def message_from(result: TurnResult, meta: str) -> dict:
    text = compose(result.headline, error_note(result), why_heading=False) if result.error else result.text
    return {"role": "assistant", "text": text, "records": result.tool_calls,
            "untraceable": result.untraceable_numbers, "meta": meta, "error": result.error,
            "grounding_issues": result.grounding_issues, "retried": result.retried}


def answer_with_agent(agent: Agent, question: str, primary: str) -> dict:
    """Typed or spoken question: the LLM chooses the tool(s), then explains."""
    started = time.monotonic()
    with st.status("Analyzing question...", expanded=False) as status:
        reporter = StatusReporter(status)
        if hasattr(agent.client, "set_sleep"):
            agent.client.set_sleep(reporter.sleep)
        result = agent.ask(question, on_event=reporter.on_event)
        done = done_line(result, time.monotonic() - started, primary)
        status.update(label=done if not result.error else "Stopped",
                      state="error" if result.error else "complete", expanded=False)
    message = message_from(result, done)
    st.session_state.messages.append(message)
    render_assistant(message, show_meta=False,     # the status label above already shows "Done"
                     key=f"msg{len(st.session_state.messages) - 1}", on_followup=ask_later)
    return message


def answer_direct(agent: Agent, question: str, tool: str, args: dict, primary: str) -> dict:
    """Example card or suggested follow-up: run the known tool now and show the computed Answer, Key facts and
    Evidence at once; then one LLM call writes the Why / drivers bullets into the slot below the Answer."""
    started = time.monotonic()
    status = st.status("Running tool...", expanded=False)
    reporter = StatusReporter(status)
    if hasattr(agent.client, "set_sleep"):
        agent.client.set_sleep(reporter.sleep)
    turn = agent.start_direct(question, tool, args, on_event=reporter.on_event)
    headline = answer_headline(turn.records)
    if headline:
        st.markdown(compose(headline, ""))
    why = st.empty()
    why.caption("Writing the explanation...")
    render_details(turn.records)
    result = agent.explain(turn, on_event=reporter.on_event)
    done = done_line(result, time.monotonic() - started, primary)
    status.update(label=done if not result.error else "Stopped",
                  state="error" if result.error else "complete", expanded=False)
    message = message_from(result, done)
    st.session_state.messages.append(message)
    with why.container():
        if result.error:
            st.markdown(error_note(result))
        else:
            st.markdown(("**Why / drivers**\n" if result.headline else "") + result.explanation)
            render_checks(message)
    if not result.error:
        render_followups(result.tool_calls, key=f"msg{len(st.session_state.messages) - 1}", on_followup=ask_later)
    return message


if request and request.get("text") and api_key:
    question = request["text"]
    agent: Agent = st.session_state.agent
    primary = getattr(agent.client, "model", "")
    first_question = not agent.turns
    key = cache_key(question, primary, PROMPT_VERSION, setup.data_info().get("built"))
    cached = setup.get_answer_cache().get(key) if first_question else None

    st.session_state.messages.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if cached is not None:                      # first question asked before: serve instantly
            agent.restore_turn(cached.turn)
            message = {**cached.message, "cached": True}
            message["meta"] = "Cached result · " + message["meta"].removeprefix("Done · ")
            st.session_state.messages.append(message)
            render_assistant(message, key=f"msg{len(st.session_state.messages) - 1}", on_followup=ask_later)
        else:
            if request.get("tool"):
                message = answer_direct(agent, question, request["tool"], request.get("args") or {}, primary)
            else:
                message = answer_with_agent(agent, question, primary)
            if (first_question and not message["error"] and not message["untraceable"]
                    and not message["grounding_issues"]):
                setup.get_answer_cache().put(key, message, agent.turns[-1])
