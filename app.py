"""Streamlit chat UI for the Airport Investment Intelligence Agent.  Run:  streamlit run app.py"""

import hashlib
import time

import streamlit as st

from agent.agent import Agent
from scoring.common import window_label
from scoring.config import WINDOW
from agent.prompts import PROMPT_VERSION
from ui import setup
from ui.cache import cache_key
from ui.progress import StatusReporter
from ui.render import EXAMPLES, about_markdown, render_assistant

st.set_page_config(page_title="Airport Investment Intelligence", page_icon=":material/flight_takeoff:",
                   layout="wide")

st.title("Airport Investment Intelligence")
st.markdown("Find US airports where terminal and capacity investments are most likely to pay off, "
            "based on passenger demand versus capacity.")

api_key = setup.get_api_key()
frames = setup.get_frames()

with st.expander("About this agent"):
    st.markdown(about_markdown(frames))

# ---------------------------------------------------------------- session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state and api_key:
    st.session_state.agent = Agent(setup.make_client(), frames)
st.session_state.setdefault("pending", None)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.subheader("Example questions")
    for i, q in enumerate(EXAMPLES):
        if st.button(q, key=f"example_{i}", width="stretch", disabled=not api_key):
            st.session_state.pending = q
    if st.button("New conversation", key="new_conversation", type="primary", width="stretch"):
        st.session_state.messages = []
        if "agent" in st.session_state:
            st.session_state.agent.reset()
        st.session_state.pop("voice_text", None)

    st.subheader("Ask by voice")
    if not api_key:
        st.caption("Available once the API key is configured.")
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
                st.session_state.pending = edited.strip()

    st.divider()
    st.caption("**Model:** " + (setup.model_info(st.session_state.agent.client)
                                if "agent" in st.session_state else "not configured"))
    info = setup.data_info()
    st.caption(f"**Data period:** {window_label(WINDOW)} (trailing 12 months)"
               + (f"  \n**Data built:** {info['built']}" if info.get("built") else ""))

# ---------------------------------------------------------------- setup notice
if not api_key:
    st.info("**Setup needed: no LLM API key found.**\n\n"
            "1. Create a free API key at your LLM provider (default: Groq, console.groq.com).\n"
            "2. Copy `.env.example` to `.env` and set `LLM_API_KEY=...` "
            "(or add `LLM_API_KEY` to Streamlit secrets when deployed).\n"
            "3. Restart the app: `streamlit run app.py`.\n\n"
            "The data and methodology above are available without a key.")

# ---------------------------------------------------------------- history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["text"])
        else:
            render_assistant(message)

# ---------------------------------------------------------------- new question
typed = st.chat_input("Ask about US airport demand, congestion or expansion candidates",
                      disabled=not api_key)
question = typed or st.session_state.pending
st.session_state.pending = None

def done_line(result, seconds: float, primary: str) -> str:
    """'Done · 1 tool call · 2,682 tokens · 3s · gpt-oss-20b (fallback)': shown once, in the status label."""
    n = len(result.tool_calls)
    names = [m.split("/")[-1] + (" (fallback)" if m != primary else "") for m in result.models]
    return (f"Done · {n} tool call{'s' if n != 1 else ''} · {result.total_tokens:,} tokens · {seconds:.0f}s"
            + (f" · {' + '.join(names)}" if names else ""))


if question and api_key:
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
            render_assistant(message)
        else:
            started = time.monotonic()
            with st.status("Analyzing question...", expanded=False) as status:
                reporter = StatusReporter(status)
                if hasattr(agent.client, "set_sleep"):
                    agent.client.set_sleep(reporter.sleep)
                result = agent.ask(question, on_event=reporter.on_event)
                done = done_line(result, time.monotonic() - started, primary)
                status.update(label=done if not result.error else "Stopped",
                              state="error" if result.error else "complete", expanded=False)
            if result.error == "rate_limited":
                text = "The LLM provider is temporarily at capacity. Please try again in about a minute."
            else:
                text = result.text
            message = {"role": "assistant", "text": text, "records": result.tool_calls,
                       "untraceable": result.untraceable_numbers, "meta": done}
            st.session_state.messages.append(message)
            render_assistant(message, show_meta=False)     # the status label above already shows "Done"
            if first_question and not result.error and not result.untraceable_numbers:
                setup.get_answer_cache().put(key, message, agent.turns[-1])
