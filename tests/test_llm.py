"""429 retry and TPM pacing with injected sleep/clock (no network, no real waiting)."""

import pytest

from agent.llm import LLMResponse, RateLimited, TpmPacer, with_retry


class Http429(Exception):
    status_code = 429

    def __init__(self, retry_after=None):
        super().__init__("rate limited")
        self.response = type("R", (), {"headers": {"retry-after": retry_after} if retry_after else {}})()


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

    with pytest.raises(RateLimited, match="rate-limited"):
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
    assert waited == pytest.approx(50.5) or waited == pytest.approx(30)   # capped by max_wait (30 s)
    assert sleeps


class _Stub:
    """Minimal stand-in for openai's client.chat.completions.with_raw_response.create(...)."""

    def __init__(self, headers):
        msg = type("M", (), {"content": "hello", "tool_calls": None})()
        usage = type("U", (), {"total_tokens": 120, "prompt_tokens": 100, "completion_tokens": 20})()
        parsed = type("C", (), {"choices": [type("Ch", (), {"message": msg})()], "usage": usage})()
        raw = type("R", (), {"headers": headers, "parse": lambda self: parsed})()
        self.kwargs = None
        outer = self

        class _Raw:
            def create(self, **kwargs):
                outer.kwargs = kwargs
                return raw
        self.chat = type("Chat", (), {"completions": type("Comp", (), {"with_raw_response": _Raw()})()})()


def test_openai_chat_complete_reads_limit_header_and_usage(monkeypatch):
    from agent.llm import OpenAIChat
    monkeypatch.delenv("LLM_TPM_LIMIT", raising=False)
    client = OpenAIChat(api_key="test", model="openai/gpt-oss-120b", reasoning_effort="low")
    client.client = _Stub({"x-ratelimit-limit-tokens": "12000"})
    resp = client.complete([{"role": "user", "content": "hi"}], tools=[])
    assert resp.content == "hello" and resp.total_tokens == 120
    assert client.pacer.limit == 12000 and client.pacer.used() == 120
    assert client.client.kwargs["reasoning_effort"] == "low"
    assert client.client.kwargs["max_completion_tokens"] == client.max_output_tokens


def test_reasoning_effort_not_sent_to_other_models(monkeypatch):
    from agent.llm import OpenAIChat
    client = OpenAIChat(api_key="test", model="qwen/qwen3.8-27b", reasoning_effort="low")
    client.client = _Stub({})
    client.complete([{"role": "user", "content": "hi"}], tools=[])
    assert "reasoning_effort" not in client.client.kwargs and client.pacer.limit == 8000


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


class _Seq:
    """Stub returning raw responses or raising per call; records the model of each call."""

    def __init__(self, outcomes):
        self.outcomes, self.models = list(outcomes), []
        outer = self

        class _Raw:
            def create(self, **kwargs):
                outer.models.append(kwargs["model"])
                item = outer.outcomes.pop(0)
                if isinstance(item, Exception):
                    raise item
                return item
        self.chat = type("Chat", (), {"completions": type("Comp", (), {"with_raw_response": _Raw()})()})()


def _raw(content="ok", tokens=100):
    msg = type("M", (), {"content": content, "tool_calls": None})()
    usage = type("U", (), {"total_tokens": tokens, "prompt_tokens": tokens, "completion_tokens": 0})()
    parsed = type("C", (), {"choices": [type("Ch", (), {"message": msg})()], "usage": usage})()
    return type("R", (), {"headers": {}, "parse": lambda self: parsed})()


def _client(**kw):
    from agent.llm import OpenAIChat
    return OpenAIChat(api_key="test", model="openai/gpt-oss-120b", fallback_model="openai/gpt-oss-20b",
                      sleep=lambda s: None, **kw)


def test_fallback_on_primary_429_without_retrying_primary():
    client = _client()
    client.client = _Seq([Http429(retry_after="40"), _raw("from fallback")])
    resp = client.complete([{"role": "user", "content": "hi"}], tools=[])
    assert resp.content == "from fallback" and resp.model == "openai/gpt-oss-20b"
    assert client.client.models == ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]


def test_fallback_when_primary_budget_would_wait_too_long():
    client = _client()
    client.pacer.record(7900)                                  # primary window nearly full
    client.client = _Seq([_raw("fast")])
    resp = client.complete([{"role": "user", "content": "hi"}], tools=[])
    assert resp.model == "openai/gpt-oss-20b"


def test_no_fallback_configured_keeps_primary():
    from agent.llm import OpenAIChat
    client = OpenAIChat(api_key="test", model="openai/gpt-oss-120b", fallback_model="", sleep=lambda s: None)
    assert client.fallback_model is None
    client.client = _Seq([_raw()])
    assert client.complete([{"role": "user", "content": "hi"}], tools=[]).model == "openai/gpt-oss-120b"
