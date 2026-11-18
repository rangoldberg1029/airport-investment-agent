"""Streamlit chat UI for the Airport Investment Intelligence Agent.  Run:  streamlit run app.py"""

import hashlib
import time

import streamlit as st

from agent.agent import Agent, TurnResult
from agent.headlines import compose
from scoring.common import window_label
from scoring.config import WINDOW
from agent.prompts import PROMPT_VERSION
from ui import setup
from ui.cache import cache_key
from ui.progress import StatusReporter, done_line
from ui.render import (
    EXAMPLE_CALLS, EXAMPLES, AnswerFrame, LiveAnswer, about_markdown, render_assistant, render_followups,
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


def send_voice() -> None:
    """Voice 'Send' callback: queue the (edited) transcript and clear the transcript box."""
    text = (st.session_state.get("voice_text") or "").strip()
    if text:
        st.session_state.pending = {"text": text}
    st.session_state.voice_text = ""


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

# A question submitted on this run, from any path: typed (chat input), card / follow-up / sidebar button or
# voice (callbacks set `pending` before the rerun). Read before anything is drawn, so the welcome cards are
# hidden at once.
typed = st.session_state.get("chat_input")
request = {"text": typed} if typed else st.session_state.pending
st.session_state.pending = None
busy = bool(request and request.get("text") and api_key)
# Empty state = no conversation yet and nothing submitted. The example questions are shown in exactly one place:
# as cards in the main area on the empty state, otherwise in the sidebar.
empty_state = not st.session_state.messages and not busy

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
            st.text_area("Transcript (check airport codes, then send)", key="voice_text", height=90)
            st.button("Send", key="voice_send", width="stretch", on_click=send_voice)

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
# Always one placeholder at this position: when a question is submitted it renders empty, which replaces the
# cards immediately instead of leaving them greyed out while the answer loads.
EXAMPLE_TITLES = ["Expansion candidates", "Congestion comparison", "Long-haul share", "Unmet demand"]
welcome = st.empty()
if empty_state:
    with welcome.container():
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
def asked_questions() -> list[str]:
    """Questions already asked in this conversation: never suggested again as follow-ups."""
    return [m["text"] for m in st.session_state.messages if m["role"] == "user"]


# Follow-up buttons only under the latest answer, and not while a new answer is loading (each element once).
messages = st.session_state.messages
last_answer = max((i for i, m in enumerate(messages) if m["role"] == "assistant"), default=None)
for idx, message in enumerate(messages):
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["text"])
        else:
            latest = idx == last_answer and not busy
            render_assistant(message, key=f"msg{idx}" if latest else None, on_followup=ask_later if latest else None,
                             asked=asked_questions())

# ---------------------------------------------------------------- new question
st.chat_input("Ask about US airport demand, congestion or expansion candidates", disabled=not api_key,
              key="chat_input")

RATE_LIMITED_NOTE = "The LLM provider is temporarily at capacity. Please try again in about a minute."


def message_from(result: TurnResult, meta: str) -> dict:
    note = RATE_LIMITED_NOTE if result.error == "rate_limited" else result.explanation
    text = compose(result.headline, note, why_heading=False) if result.error else result.text
    return {"role": "assistant", "text": text, "headline": result.headline, "explanation": note,
            "records": result.tool_calls, "untraceable": result.untraceable_numbers, "meta": meta,
            "error": result.error, "grounding_issues": result.grounding_issues, "retried": result.retried}


def answer(agent: Agent, question: str, tool: str | None, args: dict, primary: str, frame: AnswerFrame) -> dict:
    """Run one turn into `frame`. Fast path (card / follow-up: known tool + arguments): the tool runs now and the
    Answer line, Key facts and Evidence appear at once. Typed path: they appear as soon as the LLM's tool call
    has run. In both, the Why / drivers bullets then stream in under the Answer line."""
    started = time.monotonic()
    with frame.status.container():
        status = st.status("Running tool..." if tool else "Analyzing question...", expanded=False)
    reporter, live = StatusReporter(status), LiveAnswer(frame)
    if hasattr(agent.client, "set_sleep"):
        agent.client.set_sleep(reporter.sleep)

    def on_event(event: str, **kw) -> None:
        reporter.on_event(event, **kw)
        live.on_event(event, **kw)
    if tool:
        turn = agent.start_direct(question, tool, args, on_event=on_event)
        live.show_facts(turn.records, agent.headline(turn))
        result = agent.explain(turn, on_event=on_event)
    else:
        result = agent.ask(question, on_event=on_event)
    done = done_line(result, time.monotonic() - started, primary)
    status.update(label=done if not result.error else "Stopped", state="error" if result.error else "complete",
                  expanded=False)
    message = message_from(result, done)
    live.finish(message)
    return message


if busy:
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
        frame = AnswerFrame.create()
        if cached is not None:                      # first question asked before: serve instantly
            agent.restore_turn(cached.turn)
            message = {**cached.message, "cached": True}
            message["meta"] = "Cached result · " + message["meta"].removeprefix("Done · ")
        else:
            message = answer(agent, question, request.get("tool"), request.get("args") or {}, primary, frame)
            # Cache only clean first answers: no error, no retry, every check passed.
            if (first_question and not message["error"] and not message["retried"]
                    and not message["untraceable"] and not message["grounding_issues"]):
                setup.get_answer_cache().put(key, message, agent.turns[-1])
        st.session_state.messages.append(message)
        idx = len(st.session_state.messages) - 1
        if cached is not None:
            render_assistant(message, frame, key=f"msg{idx}", on_followup=ask_later, asked=asked_questions())
        elif not message["error"]:
            with frame.followups.container():
                render_followups(message["records"], f"msg{idx}", ask_later, asked_questions())
