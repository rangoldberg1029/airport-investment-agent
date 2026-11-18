"""App setup: API key lookup (.env, environment, st.secrets), cached data (auto-built if missing), LLM client.

Functions here are the seams the AppTest smoke test replaces (frames, client, key, data info).
"""

import os

import streamlit as st
from dotenv import load_dotenv

from data.db import DB_PATH
from data.repository import Frames, load_frames

SECRET_KEYS = ("GEMINI_API_KEY", "GEMINI_MODEL", "GROQ_API_KEY", "GROQ_MODEL")


def get_api_key() -> str | None:
    """The LLM key in use (GEMINI_API_KEY, else GROQ_API_KEY for Groq-only mode) from .env / environment, else
    from Streamlit secrets (Community Cloud)."""
    load_dotenv(".env")
    try:
        for key in SECRET_KEYS:
            if not os.getenv(key) and key in st.secrets:
                os.environ[key] = str(st.secrets[key])
    except Exception:          # no secrets.toml: st.secrets raises
        pass
    return os.getenv("GEMINI_API_KEY") or os.getenv("GROQ_API_KEY") or None


def voice_available() -> bool:
    """Voice input needs GROQ_API_KEY (Whisper on Groq)."""
    from agent.speech import voice_available as available

    return available()


@st.cache_resource(show_spinner=False)
def get_frames() -> Frames:
    """Load all tables once per server. If the DuckDB file is missing, build it from data/processed (offline)."""
    if not DB_PATH.exists():
        from data.etl import build

        with st.spinner("Preparing data (first run, about 20 seconds)..."):
            build("offline")
    return load_frames()


@st.cache_data(show_spinner=False)
def data_info() -> dict:
    """Data build date per source from the etl_runs table."""
    import duckdb

    try:
        with duckdb.connect(str(DB_PATH), read_only=True) as con:
            rows = con.execute("SELECT source, mode, data_period, built FROM etl_runs").fetchall()
        return {"sources": [dict(zip(("source", "mode", "period", "built"), r)) for r in rows],
                "built": max(r[3] for r in rows) if rows else None}
    except Exception:
        return {"sources": [], "built": None}


@st.cache_resource(show_spinner=False)
def get_answer_cache():
    """One answer cache per server process (first questions of a conversation only)."""
    from ui.cache import AnswerCache

    return AnswerCache()


def make_client():
    """The real LLM client: Gemini, with failover to Groq when GROQ_API_KEY is set (see agent/llm.py)."""
    from agent.llm import OpenAIChat

    return OpenAIChat.from_env()


def _host(provider) -> str:
    return getattr(provider, "base_url", "").split("//")[-1].split("/")[0]


def model_info(client) -> str:
    """Sidebar text: the model in use and the failover model, e.g.
    'gemini-3.5-flash-lite · generativelanguage.googleapis.com · reasoning: minimal' +
    'Failover: openai/gpt-oss-120b · api.groq.com' (or 'off - set GROQ_API_KEY')."""
    model = getattr(client, "model", "unknown model")
    primary = getattr(client, "primary", None)
    effort = getattr(client, "reasoning_effort", None)
    fallback = getattr(client, "fallback_model", None)
    fb = getattr(client, "fallback", None)
    text = f"{model}" + (f" · {_host(primary)}" if primary else "") + (f" · reasoning: {effort}" if effort else "")
    if fallback:
        return text + f"  \n**Failover:** {fallback}" + (f" · {_host(fb)}" if fb else "")
    if getattr(primary, "name", "") == "gemini":
        return text + "  \n**Failover:** off (set GROQ_API_KEY)"
    return text + "  \n**Failover:** none (Groq only; set GEMINI_API_KEY for the primary model)"
