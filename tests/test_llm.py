"""LLM client without network: 429 retry, pacing (provider headers or local window), the Gemini -> Groq error
policy, provider-specific request details (Gemini vs Groq), configuration from the environment."""

import pytest

from agent.llm import (
    DEFAULT_GEMINI_MODEL, DEFAULT_GROQ_MODEL, GEMINI_BASE_URL, GEMINI_PLACEHOLDER_SIGNATURE, GROQ_BASE_URL,
    SERVER_RETRY_DELAY_S, LLMResponse, OpenAIChat, RateLimited, TpmPacer, make_provider, parse_duration,
    prepare_messages, providers_from_env, with_retry,
)


class Http429(Exception):
    status_code = 429

    def __init__(self, retry_after=None):
        super().__init__("rate limited")
        self.response = type("R", (), {"headers": {"retry-after": retry_after} if retry_after else {}})()


class Http400(Exception):
    status_code = 400


# ---------------------------------------------------------------- retry

def test_retry_then_success_uses_retry_after():
    attempts, sleeps = [], []

    def fn():
        attempts.append(1)
        if len(attempts) < 3:
            raise Http429(retry_after="1.5")
        return LLMResponse("ok")

    assert with_retry(fn, sleep=sleeps.append).content == "ok"
    assert sleeps == [1.5, 1.5]


def test_retry_exhausted_raises_friendly_rate_limit():
    sleeps = []

    def fn():
        raise Http429()

    with pytest.raises(RateLimited, match="rate limit was reached"):
        with_retry(fn, max_retries=3, sleep=sleeps.append)
    assert len(sleeps) == 3 and all(s >= 2 for s in sleeps)      # backoff 2/4/8 s (+ jitter)


def test_retry_respects_total_wait_cap():
    sleeps = []
    with pytest.raises(RateLimited):
        with_retry(lambda: (_ for _ in ()).throw(Http429(retry_after="20")), max_total_wait=30, sleep=sleeps.append)
    assert sleeps == [20.0]


def test_non_rate_limit_errors_are_not_retried():
    with pytest.raises(ValueError):
        with_retry(lambda: (_ for _ in ()).throw(ValueError("bad request")), sleep=lambda s: None)


# ---------------------------------------------------------------- pacing: local window

def test_pacer_waits_until_window_has_room():
    now = [0.0]
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        now[0] += s

    pacer = TpmPacer(limit=8000, window=60, clock=lambda: now[0], sleep=sleep)
    pacer.record(7000)
    now[0] = 10.0
    assert pacer.wait_for(500) == 0                 # 7,500 fits
    waited = pacer.wait_for(2000)                    # 9,000 does not -> wait until the 7,000 expires
    assert waited == pytest.approx(30)               # capped by max_wait (30 s)
    assert sleeps


def test_pacer_window_expiry():
    now = [0.0]
    pacer = TpmPacer(limit=1000, window=60, clock=lambda: now[0], sleep=lambda s: None)
    pacer.record(900)
    now[0] = 61
    assert pacer.used() == 0


def test_wait_needed_does_not_sleep():
    now = [0.0]
    pacer = TpmPacer(limit=8000, window=60, clock=lambda: now[0], sleep=lambda s: pytest.fail("slept"))
    assert pacer.wait_needed(1000) == 0
    pacer.record(7500)
    now[0] = 20
    assert pacer.wait_needed(1000) == pytest.approx(40.5)       # until the 7,500 expires
    assert pacer.wait_needed(20000) == 60                      # bigger than the limit


def test_local_requests_per_minute_limit():
    now = [0.0]
    pacer = TpmPacer(limit=250_000, rpm_limit=2, clock=lambda: now[0], sleep=lambda s: None)
    pacer.record(100)
    now[0] = 5
    pacer.record(100)
    now[0] = 10
    assert pacer.wait_needed(100) == pytest.approx(50.5)       # 3rd request waits for the 1st to leave the window


# ---------------------------------------------------------------- pacing: provider headers

def test_parse_duration():
    assert parse_duration("7.66s") == pytest.approx(7.66)
    assert parse_duration("2m59.56s") == pytest.approx(179.56)
    assert parse_duration("120ms") == pytest.approx(0.12)
    assert parse_duration("30") == 30 and parse_duration(None) is None and parse_duration("soon") is None


def test_headers_replace_local_estimate():
    now = [0.0]
    pacer = TpmPacer(limit=8000, window=60, clock=lambda: now[0], sleep=lambda s: None)
    pacer.record(7900)                                          # local window says: almost full
    pacer.observe({"x-ratelimit-remaining-tokens": "6000", "x-ratelimit-reset-tokens": "15s"})
    assert pacer.wait_needed(5000) == 0                         # the provider says 6,000 remain
    # 2,000 missing tokens refill at 8,000/min -> 15 s (capped by the reset time)
    assert pacer.wait_needed(8000) == pytest.approx(15)
    now[0] = 10
    assert pacer.wait_needed(7000) == pytest.approx(0)          # 1,000 missing = 7.5 s, already elapsed


def test_headers_requests_exhausted_wait_for_reset():
    now = [0.0]
    pacer = TpmPacer(limit=8000, clock=lambda: now[0], sleep=lambda s: None)
    pacer.observe({"x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "2m"})
    assert pacer.wait_needed(10) == pytest.approx(120)


def test_no_rate_limit_headers_keep_local_window():
    pacer = TpmPacer(limit=1000, sleep=lambda s: None)
    pacer.observe({"content-type": "application/json"})
    assert pacer.headers is None


# ---------------------------------------------------------------- client stubs

def _raw(content="ok", tokens=100, headers=None, cached=0, tool_calls=None):
    msg = type("M", (), {"content": content, "tool_calls": tool_calls})()
    details = type("D", (), {"cached_tokens": cached})()
    usage = type("U", (), {"total_tokens": tokens, "prompt_tokens": tokens, "completion_tokens": 0,
                           "prompt_tokens_details": details})()
    parsed = type("C", (), {"choices": [type("Ch", (), {"message": msg})()], "usage": usage})()
    return type("R", (), {"headers": headers or {}, "parse": lambda self: parsed})()


class _Seq:
    """Stub for client.chat.completions.with_raw_response.create: returns or raises per call, records kwargs."""

    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), []
        outer = self

        class _Raw:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                item = outer.outcomes.pop(0)
                if isinstance(item, Exception):
                    raise item
                return item
        self.chat = type("Chat", (), {"completions": type("Comp", (), {"with_raw_response": _Raw()})()})()

    @property
    def models(self):
        return [c["model"] for c in self.calls]


def gemini(stub, **kw):
    return make_provider("gemini", DEFAULT_GEMINI_MODEL, GEMINI_BASE_URL, effort="minimal", tpm_limit=250_000,
                         rpm_limit=15, client=stub, sleep=lambda s: None, **kw)


def groq(stub, model=DEFAULT_GROQ_MODEL, **kw):
    return make_provider("groq", model, GROQ_BASE_URL, effort="low", client=stub, sleep=lambda s: None, **kw)


def client_with(primary, fallback=None):
    return OpenAIChat(primary, fallback, sleep=lambda s: None)


HI = [{"role": "user", "content": "hi"}]


# ---------------------------------------------------------------- request details per provider

def test_groq_request_reads_headers_and_excludes_cached_tokens():
    stub = _Seq([_raw("hello", tokens=120, cached=100,
                      headers={"x-ratelimit-limit-tokens": "12000", "x-ratelimit-remaining-tokens": "11000",
                               "x-ratelimit-reset-tokens": "5s"})])
    client = client_with(groq(stub))
    resp = client.complete(HI, tools=[{"type": "function"}])
    kwargs = stub.calls[0]
    assert resp.content == "hello" and resp.total_tokens == 120
    assert client.pacer.limit == 12000 and client.pacer.used() == 20         # cached prompt tokens do not count
    assert client.pacer.headers["remaining_tokens"] == 11000
    assert kwargs["reasoning_effort"] == "low" and kwargs["temperature"] == 0.2
    assert kwargs["max_completion_tokens"] == client.max_output_tokens


def test_gemini_request_uses_default_temperature_and_max_tokens():
    stub = _Seq([_raw()])
    client = client_with(gemini(stub))
    client.complete(HI, tools=[{"type": "function"}], tool_choice="none")
    kwargs = stub.calls[0]
    assert "temperature" not in kwargs and kwargs["max_tokens"] == 2000           # thinking counts toward it
    assert kwargs["reasoning_effort"] == "minimal" and kwargs["tool_choice"] == "none"


def test_reasoning_effort_not_sent_to_other_models():
    stub = _Seq([_raw()])
    client_with(make_provider("x", "qwen/qwen3.8-27b", GROQ_BASE_URL, effort="low", client=stub)).complete(HI, [])
    assert "reasoning_effort" not in stub.calls[0]


def test_rejected_reasoning_effort_is_dropped_and_call_repeated():
    stub = _Seq([Http400("Unsupported value for reasoning_effort: 'minimal'"), _raw("ok")])
    client = client_with(gemini(stub))
    assert client.complete(HI, tools=[]).content == "ok"
    assert "reasoning_effort" not in stub.calls[1] and client.reasoning_effort is None


def test_thought_signatures_gemini_vs_groq():
    history = [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "a", "type": "function", "function": {"name": "f", "arguments": "{}"},
         "extra_content": {"google": {"thought_signature": "real"}}},
        {"id": "b", "type": "function", "function": {"name": "g", "arguments": "{}"}}]}]
    g = prepare_messages(history, gemini=True)[0]
    assert [c["extra_content"]["google"]["thought_signature"] for c in g["tool_calls"]] == \
        ["real", GEMINI_PLACEHOLDER_SIGNATURE]
    assert g["content"] is None
    o = prepare_messages(history, gemini=False)[0]
    assert all("extra_content" not in c for c in o["tool_calls"])
    assert "extra_content" in history[0]["tool_calls"][0]                     # input not modified


def test_tool_call_extra_content_is_returned():
    fn = type("F", (), {"name": "rank_airports", "arguments": "{}"})()
    call = type("TC", (), {"id": "c1", "function": fn, "model_extra": {"extra_content": {"google": {"x": 1}}}})()
    client = client_with(gemini(_Seq([_raw(tool_calls=[call])])))
    resp = client.complete(HI, tools=[{"type": "function"}])
    assert resp.tool_calls[0].extra == {"google": {"x": 1}}


# ---------------------------------------------------------------- routing primary -> fallback

def test_fallback_when_primary_would_wait_too_long():
    g, q = _Seq([]), _Seq([_raw("fast")])
    client = client_with(gemini(g), groq(q))
    for _ in range(15):
        client.pacer.record(100)                                  # 15 requests this minute: RPM limit reached
    assert client.complete(HI, tools=[]).model == DEFAULT_GROQ_MODEL and g.calls == []
    assert client.switch_reason.startswith("primary rate limit (would wait")


class Http503(Exception):
    status_code = 503


class APITimeoutError(Exception):
    """Same class name as openai.APITimeoutError."""


def recording_client(primary, fallback=None):
    """Client whose waits are recorded instead of slept."""
    sleeps = []
    return OpenAIChat(primary, fallback, sleep=sleeps.append), sleeps


@pytest.mark.parametrize("error,reason", [
    (Http503("model overloaded"), "gemini server error (HTTP 503), after 1 retry"),
    (APITimeoutError("Request timed out."), "gemini timeout, after 1 retry"),
    (ConnectionError("reset"), "gemini connection error, after 1 retry"),
])
def test_5xx_or_timeout_retries_gemini_once_then_uses_groq(error, reason):
    g, q = _Seq([error, error]), _Seq([_raw("from groq")])
    client, sleeps = recording_client(gemini(g), groq(q))
    client.start_turn()
    assert client.complete(HI, tools=[]).model == DEFAULT_GROQ_MODEL
    assert g.models == [DEFAULT_GEMINI_MODEL] * 2 and sleeps == [SERVER_RETRY_DELAY_S] == [2.0]
    assert client.switch_reason == reason


def test_5xx_then_success_stays_on_gemini():
    g, q = _Seq([Http503("overloaded"), _raw("second try")]), _Seq([])
    client, sleeps = recording_client(gemini(g), groq(q))
    client.start_turn()
    resp = client.complete(HI, tools=[])
    assert resp.model == DEFAULT_GEMINI_MODEL and resp.content == "second try"
    assert sleeps == [2.0] and q.calls == [] and client.switch_reason is None


def test_429_uses_groq_immediately_without_retrying_gemini():
    g, q = _Seq([Http429(retry_after="40")]), _Seq([_raw("from groq")])
    client, sleeps = recording_client(gemini(g), groq(q))
    client.start_turn()
    assert client.complete(HI, tools=[]).model == DEFAULT_GROQ_MODEL
    assert g.models == [DEFAULT_GEMINI_MODEL] and sleeps == []
    assert client.switch_reason == "gemini rate limit (HTTP 429)"


def test_one_switch_per_turn_then_primary_again_next_turn():
    g = _Seq([Http429(), _raw("gemini again")])
    q = _Seq([_raw("1"), _raw("2")])
    client = client_with(gemini(g), groq(q))
    client.start_turn()
    client.complete(HI, tools=[])                         # primary rate-limited -> fallback
    client.complete(HI, tools=[])                         # same turn: stays on the fallback, no new primary try
    assert g.models == [DEFAULT_GEMINI_MODEL] and q.models == [DEFAULT_GROQ_MODEL] * 2
    client.start_turn()                                   # next turn starts on the primary
    assert client.complete(HI, tools=[]).model == DEFAULT_GEMINI_MODEL and client.switch_reason is None


def test_bad_request_does_not_switch_or_retry():
    g = _Seq([Http400("invalid argument")])
    client, sleeps = recording_client(gemini(g), groq(_Seq([])))
    with pytest.raises(Http400):
        client.complete(HI, tools=[])
    assert client.switch_reason is None and len(g.calls) == 1 and sleeps == []


def test_fallback_error_is_not_switched_back():
    client = client_with(gemini(_Seq([Http503("down")] * 2)), groq(_Seq([Http503("down too")] * 2)))
    with pytest.raises(Http503):
        client.complete(HI, tools=[])


def test_gemini_only_has_no_failover():
    g = _Seq([Http503("down"), Http503("still down")])
    client, sleeps = recording_client(gemini(g))
    with pytest.raises(Http503):
        client.complete(HI, tools=[])
    assert len(g.calls) == 2 and sleeps == [2.0]                 # one retry, then the error reaches the turn


def test_agent_reports_the_fallback_reason(frames):
    from agent.agent import Agent
    client = client_with(gemini(_Seq([Http503("overloaded")] * 2)), groq(_Seq([_raw("- answer")])))
    result = Agent(client, frames).ask("hello")
    assert result.models == [DEFAULT_GROQ_MODEL]
    assert result.fallback_reason == "gemini server error (HTTP 503), after 1 retry"
    from ui.progress import done_line
    assert done_line(result, 3.2, DEFAULT_GEMINI_MODEL).endswith(
        "· gpt-oss-120b (failover: gemini server error (HTTP 503), after 1 retry)")
    gemini_answer = type(result)("x", [], [], models=[DEFAULT_GEMINI_MODEL])
    assert done_line(gemini_answer, 2.0, DEFAULT_GEMINI_MODEL).endswith("· gemini-3.5-flash-lite")


def test_no_fallback_configured_keeps_primary():
    client = client_with(gemini(_Seq([_raw()])))
    assert client.fallback_model is None
    assert client.complete(HI, tools=[]).model == DEFAULT_GEMINI_MODEL


# ---------------------------------------------------------------- configuration from the environment

ENV = ("GEMINI_API_KEY", "GEMINI_MODEL", "GROQ_API_KEY", "GROQ_MODEL")


@pytest.fixture
def env(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_env_both_keys_gemini_primary_groq_failover(env):
    env.setenv("GEMINI_API_KEY", "g-key")
    env.setenv("GROQ_API_KEY", "q-key")
    primary, fallback = providers_from_env()
    assert (primary.name, primary.model, primary.base_url, primary.effort) == \
        ("gemini", DEFAULT_GEMINI_MODEL, GEMINI_BASE_URL, "minimal")
    assert primary.pacer.rpm_limit == 15 and primary.is_gemini
    assert (fallback.name, fallback.model, fallback.base_url, fallback.effort) == \
        ("groq", DEFAULT_GROQ_MODEL, GROQ_BASE_URL, "low")
    assert not fallback.is_gemini


def test_env_model_names_are_configurable(env):
    env.setenv("GEMINI_API_KEY", "g-key")
    env.setenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    env.setenv("GROQ_API_KEY", "q-key")
    env.setenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    primary, fallback = providers_from_env()
    assert primary.model == "gemini-3.1-flash-lite" and fallback.model == "qwen/qwen3.8-27b"


def test_env_gemini_only_mode_has_no_failover(env):
    env.setenv("GEMINI_API_KEY", "g-key")
    primary, fallback = providers_from_env()
    assert primary.name == "gemini" and fallback is None
    client = OpenAIChat.from_env()
    assert client.fallback_model is None
    from agent.speech import voice_available
    assert not voice_available()


def test_env_groq_only_mode_runs_on_groq(env):
    env.setenv("GROQ_API_KEY", "q-key")
    primary, fallback = providers_from_env()
    assert (primary.name, primary.model) == ("groq", DEFAULT_GROQ_MODEL) and fallback is None
    from agent.speech import voice_available
    assert voice_available()


def test_env_no_key(env):
    assert providers_from_env() == (None, None)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        OpenAIChat.from_env()
