"""App setup: API key lookup (.env, environment, st.secrets), cached data (auto-built if missing), LLM client.

Functions here are the seams the AppTest smoke test replaces (frames, client, key, data info).
"""

import os

import streamlit as st
from dotenv import load_dotenv

from data.db import DB_PATH
from data.repository import Frames, load_frames

SECRET_KEYS = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "LLM_REASONING_EFFORT")


def get_api_key() -> str | None:
    """LLM_API_KEY from .env / environment, else from Streamlit secrets (Community Cloud)."""
    load_dotenv(".env")
    try:
        for key in SECRET_KEYS:
            if not os.getenv(key) and key in st.secrets:
                os.environ[key] = str(st.secrets[key])
    except Exception:          # no secrets.toml: st.secrets raises
        pass
    return os.getenv("LLM_API_KEY") or None


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
    """The real LLM client (Groq by default; provider set via LLM_BASE_URL / LLM_MODEL)."""
    from agent.llm import OpenAIChat

    return OpenAIChat()


def model_info(client) -> str:
    model = getattr(client, "model", "unknown model")
    base = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").split("//")[-1].split("/")[0]
    effort = getattr(client, "reasoning_effort", None)
    return f"{model} · {base}" + (f" · reasoning: {effort}" if effort else "")
